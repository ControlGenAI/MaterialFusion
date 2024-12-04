import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision.utils import save_image
from torchvision.io import read_image
from diffusers import StableDiffusionPipeline
from typing import Optional, Union, Tuple, List, Callable, Dict
from einops import rearrange, repeat
from diffusion_core.p2p.utils import slerp_tensor


class MasaCtrlPipeline(StableDiffusionPipeline):
    def next_step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        x: torch.FloatTensor,
        eta=0.,
        verbose=False
    ):
        """
        Inverse sampling for DDIM Inversion
        """
        if verbose:
            print("timestep: ", timestep)
        next_step = timestep
        timestep = min(timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999)
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_step]
        beta_prod_t = 1 - alpha_prod_t
        pred_x0 = (x - beta_prod_t**0.5 * model_output) / alpha_prod_t**0.5
        pred_dir = (1 - alpha_prod_t_next)**0.5 * model_output
        x_next = alpha_prod_t_next**0.5 * pred_x0 + pred_dir
        return x_next, pred_x0

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        x: torch.FloatTensor,
        eta: float=0.0,
        verbose=False,
    ):
        """
        predict the sampe the next step in the denoise process.
        """
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep > 0 else self.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        pred_x0 = (x - beta_prod_t**0.5 * model_output) / alpha_prod_t**0.5
        pred_dir = (1 - alpha_prod_t_prev)**0.5 * model_output
        x_prev = alpha_prod_t_prev**0.5 * pred_x0 + pred_dir
        return x_prev, pred_x0

    @torch.no_grad()
    def image2latent(self, image):
        DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        if type(image) is Image:
            image = np.array(image)
            image = torch.from_numpy(image).float() / 127.5 - 1
            image = image.permute(2, 0, 1).unsqueeze(0).to(DEVICE)
        # input image density range [-1, 1]
        latents = self.vae.encode(image)['latent_dist'].mean
        latents = latents * 0.18215
        return latents

    @torch.no_grad()
    def latent2image(self, latents, return_type='np'):
        latents = 1 / 0.18215 * latents.detach()
        image = self.vae.decode(latents)['sample']
        if return_type == 'np':
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
            image = (image * 255).astype(np.uint8)
        elif return_type == "pt":
            image = (image / 2 + 0.5).clamp(0, 1)

        return image

    def latent2image_grad(self, latents):
        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents)['sample']

        return image  # range [-1, 1]

    @torch.no_grad()
    def __call__(
        self,
        prompt,
        batch_size=1,
        height=512,
        width=512,
        num_inference_steps=50,
        guidance_scale=7.5,
        eta=0.0,
        latents=None,
        unconditioning=None,
        neg_prompt=None,
        ref_intermediate_latents=None,
        return_intermediates=False,
        prox=None,
        prox_step=0,
        quantile=0.7,
        npi_interp=0,
        npi_step=0,
        **kwds
    ):
        DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        if isinstance(prompt, list):
            batch_size = len(prompt)
        elif isinstance(prompt, str):
            if batch_size > 1:
                prompt = [prompt] * batch_size
        
        if isinstance(guidance_scale, (tuple, list)):
            assert len(guidance_scale) == 2
            # guidance_scale_batch = torch.tensor(guidance_scale, device=DEVICE).reshape(2, 1, 1, 1)
            guidance_scale_0, guidance_scale_1 = guidance_scale[0], guidance_scale[1]
            guidance_scale = guidance_scale[1]
            do_separate_cfg = True
        else:
            # guidance_scale_batch = torch.tensor([guidance_scale], device=DEVICE).reshape(1, 1, 1, 1)
            do_separate_cfg = False
        
        
        # text embeddings
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            return_tensors="pt"
        )

        text_embeddings = self.text_encoder(text_input.input_ids.to(DEVICE))[0]
        if kwds.get("dir"):
            dir = text_embeddings[-2] - text_embeddings[-1]
            u, s, v = torch.pca_lowrank(dir.transpose(-1, -2), q=1, center=True)
            text_embeddings[-1] = text_embeddings[-1] + kwds.get("dir") * v


        # define initial latents
        latents_shape = (batch_size, self.unet.in_channels, height//8, width//8)
        if latents is None:
            latents = torch.randn(latents_shape, device=DEVICE)
        else:
            assert latents.shape == latents_shape, f"The shape of input latent tensor {latents.shape} should equal to predefined one."

        # unconditional embedding for classifier free guidance
        if guidance_scale > 1.:
            max_length = text_input.input_ids.shape[-1]
            if neg_prompt:
                uc_text = neg_prompt
            else:
                uc_text = ""
            # uc_text = "ugly, tiling, poorly drawn hands, poorly drawn feet, body out of frame, cut off, low contrast, underexposed, distorted face"
            if npi_interp > 0:
                assert neg_prompt is not None, "Please provide negative prompt for NPI."
                null_embedding = self.tokenizer(
                    [""] * 1,
                    padding="max_length",
                    max_length=77,
                    return_tensors="pt"
                )
                null_embedding = self.text_encoder(null_embedding.input_ids.to(DEVICE))[0]
                neg_embedding = self.tokenizer(
                    [neg_prompt] * 1,
                    padding="max_length",
                    max_length=77,
                    return_tensors="pt"
                )
                neg_embedding = self.text_encoder(neg_embedding.input_ids.to(DEVICE))[0]
                # unconditional_embeddings = (1-npi_interp) * npi_embedding + npi_interp * null_embedding
                unconditional_embeddings = slerp_tensor(npi_interp, neg_embedding, null_embedding)
                # unconditional_embeddings = unconditional_embeddings.repeat(batch_size, 1, 1)
                unconditional_embeddings = torch.cat([neg_embedding, unconditional_embeddings], dim=0)
            else:
                unconditional_input = self.tokenizer(
                    [uc_text] * batch_size,
                    padding="max_length",
                    max_length=77,
                    return_tensors="pt"
                )
                unconditional_embeddings = self.text_encoder(unconditional_input.input_ids.to(DEVICE))[0]
            # text_embeddings = torch.cat([unconditional_embeddings, text_embeddings], dim=0)
            if npi_step > 0:
                null_embedding = self.tokenizer(
                    [""] * batch_size,
                    padding="max_length",
                    max_length=77,
                    return_tensors="pt"
                )
                null_embedding = self.text_encoder(null_embedding.input_ids.to(DEVICE))[0]
                text_embeddings_null = torch.cat([null_embedding, text_embeddings], dim=0)
                text_embeddings = torch.cat([unconditional_embeddings, text_embeddings], dim=0)
            else:
                text_embeddings = torch.cat([unconditional_embeddings, text_embeddings], dim=0)


        self.scheduler.set_timesteps(num_inference_steps)
        latents_list = [latents]
        pred_x0_list = [latents]
        for i, t in enumerate(tqdm(self.scheduler.timesteps, desc="DDIM Sampler")):
            if ref_intermediate_latents is not None:
                # note that the batch_size >= 2
                latents_ref = ref_intermediate_latents[-1 - i]
                _, latents_cur = latents.chunk(2)
                latents = torch.cat([latents_ref, latents_cur])

            if guidance_scale > 1.:
                model_inputs = torch.cat([latents] * 2)
            else:
                model_inputs = latents
            if unconditioning is not None and isinstance(unconditioning, list):
                _, text_embeddings = text_embeddings.chunk(2)
                text_embeddings = torch.cat([unconditioning[i].expand(*text_embeddings.shape), text_embeddings]) 
            # predict the noise
            if npi_step >= 0 and i < npi_step:
                noise_pred = self.unet(model_inputs, t, encoder_hidden_states=text_embeddings_null).sample
            else:
                noise_pred = self.unet(model_inputs, t, encoder_hidden_states=text_embeddings).sample
            # if guidance_scale > 1.:
            #     noise_pred_uncon, noise_pred_con = noise_pred.chunk(2, dim=0)
            #     noise_pred = noise_pred_uncon + guidance_scale_batch * (noise_pred_con - noise_pred_uncon)
            
            # do CFG separately for source and target
            if do_separate_cfg:
                noise_pred_uncon, noise_pred_con = noise_pred.chunk(2, dim=0)
                noise_pred_0 = noise_pred_uncon[:batch_size//2,...] + guidance_scale_0 * (noise_pred_con[:batch_size//2,...] - noise_pred_uncon[:batch_size//2,...])                
                score_delta = noise_pred_con[batch_size//2:,...] - noise_pred_uncon[batch_size//2:,...]
                if (i >= prox_step) and (prox == 'l0' or prox == 'l1'):
                    if quantile > 0:
                        threshold = score_delta.abs().quantile(quantile)
                    else:
                        threshold = -quantile  # if quantile is negative, use it as a fixed threshold
                    # TODO: fix
                    score_delta -= score_delta.clamp(-threshold, threshold)  # hard thresholding
                    if prox == 'l1':
                        score_delta = torch.where(score_delta > 0, score_delta-threshold, score_delta)
                        score_delta = torch.where(score_delta < 0, score_delta+threshold, score_delta)
                noise_pred_1 = noise_pred_uncon[batch_size//2:,...] + guidance_scale_1 * score_delta
                noise_pred = torch.cat([noise_pred_0, noise_pred_1], dim=0)
            else:
                noise_pred_uncon, noise_pred_con = noise_pred.chunk(2, dim=0)
                noise_pred = noise_pred_uncon + guidance_scale * (noise_pred_con - noise_pred_uncon)

            # compute the previous noise sample x_t -> x_t-1
            latents, pred_x0 = self.step(noise_pred, t, latents)
            latents_list.append(latents)
            pred_x0_list.append(pred_x0)

        image = self.latent2image(latents, return_type="pt")
        if return_intermediates:
            pred_x0_list = [self.latent2image(img, return_type="pt") for img in pred_x0_list]
            latents_list = [self.latent2image(img, return_type="pt") for img in latents_list]
            return image, pred_x0_list, latents_list
        return image

    @torch.no_grad()
    def invert(
        self,
        image: torch.Tensor,
        prompt,
        num_inference_steps=50,
        guidance_scale=7.5,
        eta=0.0,
        return_intermediates=False,
        **kwds
    ):
        """
        invert a real image into noise map with determinisc DDIM inversion
        """
        DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        batch_size = image.shape[0]
        if isinstance(prompt, list):
            if batch_size == 1:
                image = image.expand(len(prompt), -1, -1, -1)
        elif isinstance(prompt, str):
            if batch_size > 1:
                prompt = [prompt] * batch_size

        # text embeddings
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            return_tensors="pt"
        )
        text_embeddings = self.text_encoder(text_input.input_ids.to(DEVICE))[0]
        
        # define initial latents
        latents = self.image2latent(image)
        start_latents = latents
        
        # unconditional embedding for classifier free guidance
        if guidance_scale > 1.:
            max_length = text_input.input_ids.shape[-1]
            unconditional_input = self.tokenizer(
                [""] * batch_size,
                padding="max_length",
                max_length=77,
                return_tensors="pt"
            )
            unconditional_embeddings = self.text_encoder(unconditional_input.input_ids.to(DEVICE))[0]
            text_embeddings = torch.cat([unconditional_embeddings, text_embeddings], dim=0)

        # print("latents shape: ", latents.shape)
        # interative sampling
        self.scheduler.set_timesteps(num_inference_steps)
        # print("Valid timesteps: ", reversed(self.scheduler.timesteps))
        # print("attributes: ", self.scheduler.__dict__)
        latents_list = [latents]
        pred_x0_list = [latents]
        for i, t in enumerate(tqdm(reversed(self.scheduler.timesteps), desc="DDIM Inversion", disable=True)):
            if guidance_scale > 1.:
                model_inputs = torch.cat([latents] * 2)
            else:
                model_inputs = latents

            # predict the noise
            noise_pred = self.unet(model_inputs, t, encoder_hidden_states=text_embeddings).sample
            if guidance_scale > 1.:
                noise_pred_uncon, noise_pred_con = noise_pred.chunk(2, dim=0)
                noise_pred = noise_pred_uncon + guidance_scale * (noise_pred_con - noise_pred_uncon)
            # compute the previous noise sample x_t-1 -> x_t
            latents, pred_x0 = self.next_step(noise_pred, t, latents)
            latents_list.append(latents)
            pred_x0_list.append(pred_x0)

        if return_intermediates:
            # return the intermediate laters during inversion
            # pred_x0_list = [self.latent2image(img, return_type="pt") for img in pred_x0_list]
            return latents, latents_list
        return latents, start_latents


class AttentionBase:
    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0

    def after_step(self):
        pass

    def __call__(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        out = self.forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            # after step
            self.after_step()
        return out

    def forward(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        out = torch.einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=num_heads)
        return out

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0


class AttentionStore(AttentionBase):
    def __init__(self, res=[32], min_step=0, max_step=1000):
        super().__init__()
        self.res = res
        self.min_step = min_step
        self.max_step = max_step
        self.valid_steps = 0

        self.self_attns = []  # store the all attns
        self.cross_attns = []

        self.self_attns_step = []  # store the attns in each step
        self.cross_attns_step = []

    def after_step(self):
        if self.cur_step > self.min_step and self.cur_step < self.max_step:
            self.valid_steps += 1
            if len(self.self_attns) == 0:
                self.self_attns = self.self_attns_step
                self.cross_attns = self.cross_attns_step
            else:
                for i in range(len(self.self_attns)):
                    self.self_attns[i] += self.self_attns_step[i]
                    self.cross_attns[i] += self.cross_attns_step[i]
        self.self_attns_step.clear()
        self.cross_attns_step.clear()

    def forward(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        if attn.shape[1] <= 64 ** 2:  # avoid OOM
            if is_cross:
                self.cross_attns_step.append(attn)
            else:
                self.self_attns_step.append(attn)
        return super().forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)


def register_attention_editor_diffusers(model, editor: AttentionBase):
    """
    Register a attention editor to Diffuser Pipeline, refer from [Prompt-to-Prompt]
    """
    def ca_forward(self, place_in_unet):
        def forward(x, encoder_hidden_states=None, attention_mask=None, context=None, mask=None):
            """
            The attention is similar to the original implementation of LDM CrossAttention class
            except adding some modifications on the attention
            """
            if encoder_hidden_states is not None:
                context = encoder_hidden_states
            if attention_mask is not None:
                mask = attention_mask

            to_out = self.to_out
            if isinstance(to_out, nn.modules.container.ModuleList):
                to_out = self.to_out[0]
            else:
                to_out = self.to_out

            h = self.heads
            q = self.to_q(x)
            is_cross = context is not None
            context = context if is_cross else x
            k = self.to_k(context)
            v = self.to_v(context)
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

            sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale

            if mask is not None:
                mask = rearrange(mask, 'b ... -> b (...)')
                max_neg_value = -torch.finfo(sim.dtype).max
                mask = repeat(mask, 'b j -> (b h) () j', h=h)
                mask = mask[:, None, :].repeat(h, 1, 1)
                sim.masked_fill_(~mask, max_neg_value)

            attn = sim.softmax(dim=-1)
            # the only difference
            out = editor(
                q, k, v, sim, attn, is_cross, place_in_unet,
                self.heads, scale=self.scale)

            return to_out(out)

        return forward

    def register_editor(net, count, place_in_unet):
        for name, subnet in net.named_children():
            if net.__class__.__name__ == 'Attention':  # spatial Transformer layer
                net.forward = ca_forward(net, place_in_unet)
                return count + 1
            elif hasattr(net, 'children'):
                count = register_editor(subnet, count, place_in_unet)
        return count

    cross_att_count = 0
    for net_name, net in model.unet.named_children():
        if "down" in net_name:
            cross_att_count += register_editor(net, 0, "down")
        elif "mid" in net_name:
            cross_att_count += register_editor(net, 0, "mid")
        elif "up" in net_name:
            cross_att_count += register_editor(net, 0, "up")
    editor.num_att_layers = cross_att_count


def regiter_attention_editor_ldm(model, editor: AttentionBase):
    """
    Register a attention editor to Stable Diffusion model, refer from [Prompt-to-Prompt]
    """
    def ca_forward(self, place_in_unet):
        def forward(x, encoder_hidden_states=None, attention_mask=None, context=None, mask=None):
            """
            The attention is similar to the original implementation of LDM CrossAttention class
            except adding some modifications on the attention
            """
            if encoder_hidden_states is not None:
                context = encoder_hidden_states
            if attention_mask is not None:
                mask = attention_mask

            to_out = self.to_out
            if isinstance(to_out, nn.modules.container.ModuleList):
                to_out = self.to_out[0]
            else:
                to_out = self.to_out

            h = self.heads
            q = self.to_q(x)
            is_cross = context is not None
            context = context if is_cross else x
            k = self.to_k(context)
            v = self.to_v(context)
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

            sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale

            if mask is not None:
                mask = rearrange(mask, 'b ... -> b (...)')
                max_neg_value = -torch.finfo(sim.dtype).max
                mask = repeat(mask, 'b j -> (b h) () j', h=h)
                mask = mask[:, None, :].repeat(h, 1, 1)
                sim.masked_fill_(~mask, max_neg_value)

            attn = sim.softmax(dim=-1)
            # the only difference
            out = editor(
                q, k, v, sim, attn, is_cross, place_in_unet,
                self.heads, scale=self.scale)

            return to_out(out)

        return forward

    def register_editor(net, count, place_in_unet):
        for name, subnet in net.named_children():
            if net.__class__.__name__ == 'CrossAttention':  # spatial Transformer layer
                net.forward = ca_forward(net, place_in_unet)
                return count + 1
            elif hasattr(net, 'children'):
                count = register_editor(subnet, count, place_in_unet)
        return count

    cross_att_count = 0
    for net_name, net in model.model.diffusion_model.named_children():
        if "input" in net_name:
            cross_att_count += register_editor(net, 0, "input")
        elif "middle" in net_name:
            cross_att_count += register_editor(net, 0, "middle")
        elif "output" in net_name:
            cross_att_count += register_editor(net, 0, "output")
    editor.num_att_layers = cross_att_count

    

class MutualSelfAttentionControl(AttentionBase):
    MODEL_TYPE = {
        "SD": 16,
        "SDXL": 70
    }

    def __init__(self, start_step=4, start_layer=10, layer_idx=None, step_idx=None, total_steps=50, model_type="SD"):
        """
        Mutual self-attention control for Stable-Diffusion model
        Args:
            start_step: the step to start mutual self-attention control
            start_layer: the layer to start mutual self-attention control
            layer_idx: list of the layers to apply mutual self-attention control
            step_idx: list the steps to apply mutual self-attention control
            total_steps: the total number of steps
            model_type: the model type, SD or SDXL
        """
        super().__init__()
        self.total_steps = total_steps
        self.total_layers = self.MODEL_TYPE.get(model_type, 16)
        self.start_step = start_step
        self.start_layer = start_layer
        self.layer_idx = layer_idx if layer_idx is not None else list(range(start_layer, self.total_layers))
        self.step_idx = step_idx if step_idx is not None else list(range(start_step, total_steps))
        print("MasaCtrl at denoising steps: ", self.step_idx)
        print("MasaCtrl at U-Net layers: ", self.layer_idx)

    def attn_batch(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        """
        Performing attention for a batch of queries, keys, and values
        """
        b = q.shape[0] // num_heads
        q = rearrange(q, "(b h) n d -> h (b n) d", h=num_heads)
        k = rearrange(k, "(b h) n d -> h (b n) d", h=num_heads)
        v = rearrange(v, "(b h) n d -> h (b n) d", h=num_heads)

        sim = torch.einsum("h i d, h j d -> h i j", q, k) * kwargs.get("scale")
        attn = sim.softmax(-1)
        out = torch.einsum("h i j, h j d -> h i d", attn, v)
        out = rearrange(out, "h (b n) d -> b n (h d)", b=b)
        return out

    def forward(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        """
        Attention forward function
        """
        if is_cross or self.cur_step not in self.step_idx or self.cur_att_layer // 2 not in self.layer_idx:
            return super().forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)

        qu, qc = q.chunk(2)
        ku, kc = k.chunk(2)
        vu, vc = v.chunk(2)
        attnu, attnc = attn.chunk(2)

        out_u = self.attn_batch(qu, ku[:num_heads], vu[:num_heads], sim[:num_heads], attnu, is_cross, place_in_unet, num_heads, **kwargs)
        out_c = self.attn_batch(qc, kc[:num_heads], vc[:num_heads], sim[:num_heads], attnc, is_cross, place_in_unet, num_heads, **kwargs)
        out = torch.cat([out_u, out_c], dim=0)

        return out

    
class MutualSelfAttentionControlProx(AttentionBase):
    def __init__(self, start_step=4, start_layer=10, layer_idx=None, step_idx=None, total_steps=50, inject_uncond="src", inject_cond="src"):
        """
        Mutual self-attention control for Stable-Diffusion model
        Args:
            start_step: the step to start mutual self-attention control
            start_layer: the layer to start mutual self-attention control
            layer_idx: list of the layers to apply mutual self-attention control
            step_idx: list the steps to apply mutual self-attention control
            total_steps: the total number of steps
        """
        super().__init__()
        self.total_steps = total_steps
        self.start_step = start_step
        self.start_layer = start_layer
        self.layer_idx = layer_idx if layer_idx is not None else list(range(start_layer, 16))
        self.step_idx = step_idx if step_idx is not None else list(range(start_step, total_steps))
        self.inject_uncond = inject_uncond
        self.inject_cond = inject_cond
        print("step_idx: ", self.step_idx)
        print("layer_idx: ", self.layer_idx)

    def attn_batch(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        b = q.shape[0] // num_heads
        q = rearrange(q, "(b h) n d -> h (b n) d", h=num_heads)
        k = rearrange(k, "(b h) n d -> h (b n) d", h=num_heads)
        v = rearrange(v, "(b h) n d -> h (b n) d", h=num_heads)

        sim = torch.einsum("h i d, h j d -> h i j", q, k) * kwargs.get("scale")
        attn = sim.softmax(-1)
        out = torch.einsum("h i j, h j d -> h i d", attn, v)
        out = rearrange(out, "h (b n) d -> b n (h d)", b=b)
        return out

    def forward(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        """
        Attention forward function
        """
        if is_cross or self.cur_step not in self.step_idx or self.cur_att_layer // 2 not in self.layer_idx:
            return super().forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)

        qu, qc = q.chunk(2)
        ku, kc = k.chunk(2)
        vu, vc = v.chunk(2)
        attnu, attnc = attn.chunk(2)

        # if self.inject_uncond == "src":
        #     out_u = self.attn_batch(qu, ku[:num_heads], vu[:num_heads], None, attnu, is_cross, place_in_unet, num_heads, **kwargs)
        # elif self.inject_uncond == "joint":
        #     out_u = self.attn_batch(qu, ku, vu, None, attnu, is_cross, place_in_unet, num_heads, **kwargs)
        # elif self.inject_uncond == "none":  # no swap
        #     out_u = torch.cat([
        #         self.attn_batch(qu[:num_heads], ku[:num_heads], vu[:num_heads], None, attnu, is_cross, place_in_unet, num_heads, **kwargs),
        #         self.attn_batch(qu[num_heads:], ku[num_heads:], vu[num_heads:], None, attnu, is_cross, place_in_unet, num_heads, **kwargs)], dim=0)
        # elif self.inject_uncond == "tar":  # this should never be used
        #     out_u = self.attn_batch(qu, ku[num_heads:], vu[num_heads:], None, attnu, is_cross, place_in_unet, num_heads, **kwargs)
        # else:
        #     raise NotImplementedError
        # if self.inject_cond == "src":
        #     out_c = self.attn_batch(qc, kc[:num_heads], vc[:num_heads], None, attnc, is_cross, place_in_unet, num_heads, **kwargs)
        # elif self.inject_cond == "joint":
        #     out_c = self.attn_batch(qc, kc, vc, None, attnc, is_cross, place_in_unet, num_heads, **kwargs)
        # elif self.inject_cond == "none":  # no swap
        #     out_c = torch.cat([
        #         self.attn_batch(qc[:num_heads], kc[:num_heads], vc[:num_heads], None, attnc, is_cross, place_in_unet, num_heads, **kwargs),
        #         self.attn_batch(qc[num_heads:], kc[num_heads:], vc[num_heads:], None, attnc, is_cross, place_in_unet, num_heads, **kwargs)], dim=0)
        # elif self.inject_cond == "tar":  # this should never be used
        #     out_c = self.attn_batch(qc, kc[num_heads:], vc[num_heads:], None, attnc, is_cross, place_in_unet, num_heads, **kwargs)
        # else:
        #     raise NotImplementedError
        # out = torch.cat([out_u, out_c], dim=0)

        out_u_0 = self.attn_batch(qu[:num_heads], ku[:num_heads], vu[:num_heads], None, attnu, is_cross, place_in_unet, num_heads, **kwargs)
        out_c_0 = self.attn_batch(qc[:num_heads], kc[:num_heads], vc[:num_heads], None, attnc, is_cross, place_in_unet, num_heads, **kwargs)
        if self.inject_uncond == "src":
            out_u_1 = self.attn_batch(qu[num_heads:], ku[:num_heads], vu[:num_heads], None, attnu, is_cross, place_in_unet, num_heads, **kwargs)
        elif self.inject_uncond == "joint":
            out_u_1 = self.attn_batch(qu[num_heads:], ku, vu, None, attnu, is_cross, place_in_unet, num_heads, **kwargs)
        elif self.inject_uncond == "none" or self.inject_uncond == "tar":  # no swap
            out_u_1 = self.attn_batch(qu[num_heads:], ku[num_heads:], vu[num_heads:], None, attnu, is_cross, place_in_unet, num_heads, **kwargs)
        else:
            raise NotImplementedError
        if self.inject_cond == "src":
            out_c_1 = self.attn_batch(qc[num_heads:], kc[:num_heads], vc[:num_heads], None, attnc, is_cross, place_in_unet, num_heads, **kwargs)
        elif self.inject_cond == "joint":
            out_c_1 = self.attn_batch(qc[num_heads:], kc, vc, None, attnc, is_cross, place_in_unet, num_heads, **kwargs)
        elif self.inject_cond == "none" or self.inject_cond == "tar":  # no swap
            out_c_1 = self.attn_batch(qc[num_heads:], kc[num_heads:], vc[num_heads:], None, attnc, is_cross, place_in_unet, num_heads, **kwargs)
        else:
            raise NotImplementedError
        out = torch.cat([out_u_0, out_u_1, out_c_0, out_c_1], dim=0)

        return out
