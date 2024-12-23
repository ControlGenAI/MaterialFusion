import json
from collections import OrderedDict
from typing import Callable, Dict, Optional
import torch
import numpy as np
import PIL
import gc
from typing import Optional, Union, Tuple, List, Callable, Dict
from tqdm.auto import trange, tqdm
from diffusion_core.guiders.opt_guiders import opt_registry
from diffusion_core.diffusion_utils import latent2image, image2latent
from diffusion_core.custom_forwards.unet_sd import unet_forward
from diffusion_core.guiders.noise_rescales import noise_rescales
from diffusion_core.inversion import Inversion, NullInversion, NegativePromptInversion
from diffusion_core.utils import toggle_grad, use_grad_checkpointing, is_torch2_available, get_generator
import os
from safetensors import safe_open
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from diffusers.pipelines.controlnet import MultiControlNetModel
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import torchvision.utils as vutils
from diffusers.image_processor import IPAdapterMaskProcessor

if is_torch2_available():
    from ip_adapter.attention_processor import (
        AttnProcessor2_0 as AttnProcessor
    )
    from ip_adapter.attention_processor import (
        CNAttnProcessor2_0 as CNAttnProcessor
    )
    from ip_adapter.attention_processor import (
        IPAttnProcessor2_0 as IPAttnProcessor,
    )
else:
    from ip_adapter.attention_processor import AttnProcessor, CNAttnProcessor, IPAttnProcessor


class ImageProjModel(torch.nn.Module):
    """Projection Model"""

    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
        super().__init__()

        self.generator = None
        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = torch.nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds):
        embeds = image_embeds
        clip_extra_context_tokens = self.proj(embeds).reshape(
            -1, self.clip_extra_context_tokens, self.cross_attention_dim
        )
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens


class GuidanceEditing:
    def __init__(
            self,
            model,
            config,
            image_encoder_path,
            ip_ckpt, 
            device,
            num_tokens=4
    ):

        self.config = config
        self.model = model
  
        toggle_grad(self.model.unet, False)

        if config.get('gradient_checkpointing', False):
            use_grad_checkpointing(mode=True)
        else:
            use_grad_checkpointing(mode=False)

        self.guiders = {
            g_data.name: (opt_registry[g_data.name](**g_data.get('kwargs', {})), g_data.g_scale)
            for g_data in config.guiders
        }

        self._setup_inversion_engine()
        self.latents_stack = []

        self.context = None

        self.noise_rescaler = noise_rescales[config.noise_rescaling_setup.type](
            config.noise_rescaling_setup.init_setup,
            **config.noise_rescaling_setup.get('kwargs', {})
        )

        for guider_name, (guider, _) in self.guiders.items():
            guider.clear_outputs()
        
        self.self_attn_layers_num = config.get('self_attn_layers_num', [6, 1, 9])
        if type(self.self_attn_layers_num[0]) is int:
            for i in range(len(self.self_attn_layers_num)):
                self.self_attn_layers_num[i] = (0, self.self_attn_layers_num[i])
        #ip-part
        self.device = device
        self.image_encoder_path = image_encoder_path
        self.ip_ckpt = ip_ckpt
        self.num_tokens = num_tokens
        self.set_ip_adapter()
        print("ip-adapter is set")
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(self.image_encoder_path).to(
            self.device
        )
        self.clip_image_processor = CLIPImageProcessor()
        self.image_proj_model = self.init_proj()
        self.load_ip_adapter()
        print("ip-adapter is loaded")

    
    def set_ip_adapter(self):
        unet = self.model.unet
        attn_procs = {}
        for name in unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            if cross_attention_dim is None:
                attn_procs[name] = AttnProcessor()
            else:
                attn_procs[name] = IPAttnProcessor(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    scale=1.0,
                    num_tokens=self.num_tokens
                ).to(self.device)
                
        unet.set_attn_processor(attn_procs)
        if hasattr(self.model, "controlnet"):
            if isinstance(self.model.controlnet, MultiControlNetModel):
                for controlnet in self.model.controlnet.nets:
                    controlnet.set_attn_processor(CNAttnProcessor(num_tokens=self.num_tokens))
            else:
                self.model.controlnet.set_attn_processor(CNAttnProcessor(num_tokens=self.num_tokens))


    def init_proj(self):
        image_proj_model = ImageProjModel(
            cross_attention_dim=self.model.unet.config.cross_attention_dim,
            clip_embeddings_dim=self.image_encoder.config.projection_dim,
            clip_extra_context_tokens=self.num_tokens
        ).to(self.device)
        return image_proj_model
    

    def load_ip_adapter(self):
        if os.path.splitext(self.ip_ckpt)[-1] == ".safetensors":
            state_dict = {"image_proj": {}, "ip_adapter": {}}
            with safe_open(self.ip_ckpt, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key.startswith("image_proj."):
                        state_dict["image_proj"][key.replace("image_proj.", "")] = f.get_tensor(key)
                    elif key.startswith("ip_adapter."):
                        state_dict["ip_adapter"][key.replace("ip_adapter.", "")] = f.get_tensor(key)
        else:
            state_dict = torch.load(self.ip_ckpt, map_location="cpu")
        self.image_proj_model.load_state_dict(state_dict["image_proj"])
        ip_layers = torch.nn.ModuleList(self.model.unet.attn_processors.values())
        ip_layers.load_state_dict(state_dict["ip_adapter"])


    def _setup_inversion_engine(self):
        if self.config.inversion_type == 'ntinv':
            self.inversion_engine = NullInversion(
                self.model,
                self.model.scheduler.num_inference_steps,
                self.config.guiders[0]['g_scale'],
                forward_guidance_scale=1,
                verbose=self.config.verbose
            )
        elif self.config.inversion_type == 'npinv':
            self.inversion_engine = NegativePromptInversion(
                self.model,
                self.model.scheduler.num_inference_steps,
                self.config.guiders[0]['g_scale'],
                forward_guidance_scale=1,
                verbose=self.config.verbose
            )
        elif self.config.inversion_type == 'dummy':
            self.inversion_engine = Inversion(
                self.model,
                self.model.scheduler.num_inference_steps,
                self.config.guiders[0]['g_scale'],
                forward_guidance_scale=1,
                verbose=self.config.verbose
            )
        else:
            raise ValueError('Incorrect InversionType')

    def set_scale_ip(self, scale):
        for attn_processor in self.model.unet.attn_processors.values():
            if isinstance(attn_processor, IPAttnProcessor):
                attn_processor.scale = scale
                attn_processor.full_emb_size = scale

    def set_full_emb_size(self, full_emb_size):
        for attn_processor in self.model.unet.attn_processors.values():
            if isinstance(attn_processor, IPAttnProcessor):
                attn_processor.full_emb_size = full_emb_size

    def generate_embeds_ip(
        self,
        pil_image=None,
        clip_image_embeds=None,
        prompt=None,
        negative_prompt=None,
        scale=0.0,   
        num_samples=1
    ):
        self.set_scale_ip(scale)

        if pil_image is not None:
            num_prompts = 1 if isinstance(pil_image, Image.Image) else len(pil_image)
        else:
            num_prompts = clip_image_embeds.size(0)

        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts
        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds_ip(
            pil_image=pil_image, clip_image_embeds=clip_image_embeds
        )
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

        with torch.inference_mode():
            prompt_embeds_, negative_prompt_embeds_ = self.model.encode_prompt(
                prompt,
                device=self.device,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            prompt_embeds = torch.cat([prompt_embeds_, image_prompt_embeds], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds_, uncond_image_prompt_embeds], dim=1)
        self.prompt_embeds = prompt_embeds
        self.negative_prompt_embeds = negative_prompt_embeds
        self.set_full_emb_size(prompt_embeds.shape[1])


    @torch.inference_mode()
    def get_image_embeds_ip(self, pil_image=None, clip_image_embeds=None):
        if pil_image is not None:
            if isinstance(pil_image, Image.Image):
                pil_image = [pil_image]
            clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
            clip_image_embeds = self.image_encoder(clip_image.to(self.device)).image_embeds
        else:
            clip_image_embeds = clip_image_embeds.to(self.device) 
        image_prompt_embeds = self.image_proj_model(clip_image_embeds)
        uncond_image_prompt_embeds = self.image_proj_model(torch.zeros_like(clip_image_embeds))
        return image_prompt_embeds, uncond_image_prompt_embeds


    def __call__(
            self,
            image_gt: PIL.Image.Image,
            inv_prompt: str,
            trg_prompt: str,
            ip_image: PIL.Image.Image,
            scale = 0.0,
            control_image: Optional[PIL.Image.Image] = None,
            verbose: bool = False,
            background_mask = None,
            cross_attention_kwargs = None
    ):
        self.train(
            image_gt,
            inv_prompt,
            trg_prompt,
            ip_image,
            scale,
            control_image,
            verbose,
            background_mask,
            cross_attention_kwargs
        )

        return self.edit()

    def train(
            self,
            image_gt: PIL.Image.Image,
            inv_prompt: str,
            trg_prompt: str,
            ip_image: PIL.Image.Image,
            scale,
            control_image: Optional[PIL.Image.Image] = None,
            verbose: bool = False, 
            background_mask = None,
            cross_attention_kwargs=None
    ):
        self.background_mask = background_mask
        self.cross_attention_kwargs = cross_attention_kwargs
        self.init_prompt(inv_prompt, trg_prompt)
        self.generate_embeds_ip(pil_image=ip_image, prompt=trg_prompt,scale=scale)
        self.verbose = verbose

        image_gt = np.array(image_gt)
        if self.config.start_latent == 'inversion':
            _, self.inv_latents, self.uncond_embeddings = self.inversion_engine(
                image_gt, inv_prompt,
                verbose=self.verbose
            )
        elif self.config.start_latent == 'random':
            self.inv_latents = self.sample_noised_latents(
                image2latent(image_gt, self.model)
            )
        else:
            raise ValueError('Incorrect start latent type')
        
        for g_name, (guider, _) in self.guiders.items():
            if hasattr(guider, 'model_patch'):
                guider.model_patch(self.model, self_attn_layers_num=self.self_attn_layers_num)

        self.start_latent = self.inv_latents[-1].clone()

        params = {
            'model': self.model,
            'inv_prompt': inv_prompt,
            'trg_prompt': trg_prompt
        }

        for g_name, (guider, _) in self.guiders.items():
            if hasattr(guider, 'train'):
                guider.train(params)

        for guider_name, (guider, _) in self.guiders.items():
            guider.clear_outputs()

    def _construct_data_dict(
            self, latents,
            diffusion_iter,
            timestep
    ):
        uncond_emb, inv_prompt_emb, trg_prompt_emb = self.context.chunk(3)
        negative_prompt_embeds_ip = self.negative_prompt_embeds
        trg_prompt_emb = self.prompt_embeds 
        if self.uncond_embeddings is not None:
            uncond_emb = self.uncond_embeddings[diffusion_iter]

        data_dict = {
            'latent': latents,
            'inv_latent': self.inv_latents[-diffusion_iter - 1],
            'timestep': timestep,
            'model': self.model,
            'uncond_emb': uncond_emb,
            'trg_emb': trg_prompt_emb,
            'inv_emb': inv_prompt_emb,
            'diff_iter': diffusion_iter
        }
        with torch.no_grad():
            uncond_unet = unet_forward(
                self.model,
                data_dict['latent'],
                data_dict['timestep'],
                data_dict['uncond_emb'],
                None
            )

        for g_name, (guider, _) in self.guiders.items():
            if hasattr(guider, 'model_patch'):
                guider.clear_outputs()
        with torch.no_grad():
            inv_prompt_unet = unet_forward(
                self.model,
                data_dict['inv_latent'],
                data_dict['timestep'],
                data_dict['inv_emb'],
                None
            )

        for g_name, (guider, _) in self.guiders.items():
            if hasattr(guider, 'model_patch'):
                if 'inv_inv' in guider.forward_hooks:
                    data_dict.update({f"{g_name}_inv_inv": guider.output})
                guider.clear_outputs()

        data_dict['latent'].requires_grad = True
        src_prompt_unet = unet_forward(
            self.model,
            data_dict['latent'],
            data_dict['timestep'],
            data_dict['inv_emb'],
            None
        )

        for g_name, (guider, _) in self.guiders.items(): 
            if hasattr(guider, 'model_patch'):
                if 'cur_inv' in guider.forward_hooks:
                    data_dict.update({f"{g_name}_cur_inv": guider.output})
                guider.clear_outputs()
        
        trg_prompt_unet = unet_forward(
            self.model,
            data_dict['latent'],
            data_dict['timestep'],
            data_dict['trg_emb'],
            None,
            cross_attention_kwargs = self.cross_attention_kwargs
        )

        for g_name, (guider, _) in self.guiders.items():
            if hasattr(guider, 'model_patch'):
                if 'cur_trg' in guider.forward_hooks:
                    data_dict.update({f"{g_name}_cur_trg": guider.output})
                guider.clear_outputs()

        data_dict.update({
            'uncond_unet': uncond_unet,
            'trg_prompt_unet': trg_prompt_unet,
        })

        return data_dict

    def _get_noise(self, data_dict, diffusion_iter):
        backward_guiders_sum = 0.
        noises = {
            'uncond': data_dict['uncond_unet'],
        }
        index = torch.where(self.model.scheduler.timesteps == data_dict['timestep'])[0].item()

        for name, (guider, g_scale) in self.guiders.items():
            if guider.grad_guider:
                cur_noise_pred = self._get_scale(g_scale, diffusion_iter) * guider(data_dict)
                noises[name] = cur_noise_pred
            else:
                energy = self._get_scale(g_scale, diffusion_iter) * guider(data_dict)
                comparison_tensor = torch.tensor(0.).to(energy.dtype)
                if not torch.allclose(energy, comparison_tensor):
                    backward_guiders_sum += energy
        
        if hasattr(backward_guiders_sum, 'backward'):
            backward_guiders_sum.backward()
            noises['other'] = data_dict['latent'].grad
        scales = self.noise_rescaler(noises, index)
        noise_pred = sum(scales[k] * noises[k] for k in noises)
        for g_name, (guider, _) in self.guiders.items():
            if not guider.grad_guider:
                guider.clear_outputs()
            gc.collect()
            torch.cuda.empty_cache()

        return noise_pred

    @staticmethod
    def _get_scale(g_scale, diffusion_iter):
        if type(g_scale) is float:
            return g_scale
        else:
            return g_scale[diffusion_iter]

    @torch.no_grad()
    def _step(self, noise_pred, t, latents):
        latents = self.model.scheduler.step_backward(noise_pred, t, latents).prev_sample
        self.latents_stack.append(latents.detach())
        return latents

    def edit(self):
        self.model.scheduler.set_timesteps(self.model.scheduler.num_inference_steps)
        latents = self.start_latent
        self.latents_stack = []

        for i, timestep in tqdm(
                enumerate(self.model.scheduler.timesteps),
                total=self.model.scheduler.num_inference_steps,
                desc='Editing',
                disable=not self.verbose
        ):
            # 1. Construct dict            
            data_dict = self._construct_data_dict(latents, i, timestep)

            # 2. Calculate guidance
            noise_pred = self._get_noise(data_dict, i)

            # 3. Scheduler step
            latents = self._step(noise_pred, timestep, latents) #torch.Size([1, 4, 64, 64])

            # добавим маскирование: на каждой итерации будем брать фон с z_t^*, а объект с z_t, используя маски
            if self.background_mask is not None:
                if i < self.background_mask-1: # с 0 по 49
                    # Step 1: Downsample the mask from 512x512 to 64x64 using interpolation
                    mask_downsampled = F.interpolate(self.cross_attention_kwargs["ip_adapter_masks"][0].float(), size=(64, 64), mode='bilinear')
                    # Step 2: Broadcast the mask to have 4 channels (same as the latents)
                    mask_broadcasted = mask_downsampled.repeat(1, 4, 1, 1) # Shape: (1, 4, 64, 64)
                    mask_broadcasted = mask_broadcasted.to(self.device)
                    # Step 3: Combine latents using the mask
                    # Where mask == 1, take from latent2 (object), and where mask == 0, take from latent1 (background)
                    latents = self.inv_latents[-i - 1] * (1 - mask_broadcasted) + latents * mask_broadcasted
            else:
                print('background mask is None')

        self._model_unpatch(self.model)
        return latent2image(latents, self.model)[0]

    @torch.no_grad()
    def init_prompt(self, inv_prompt: str, trg_prompt: str):
        trg_prompt_embed = self.get_prompt_embed(trg_prompt)
        inv_prompt_embed = self.get_prompt_embed(inv_prompt)
        uncond_embed = self.get_prompt_embed("")

        self.context = torch.cat([uncond_embed, inv_prompt_embed, trg_prompt_embed])

    def get_prompt_embed(self, prompt: str):
        text_input = self.model.tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.model.text_encoder(
            text_input.input_ids.to(self.model.device)
        )[0]

        return text_embeddings

    def sample_noised_latents(self, latent):
        all_latent = [latent.clone().detach()]
        latent = latent.clone().detach()
        for i in trange(self.model.scheduler.num_inference_steps, desc='Latent Sampling'):
            timestep = self.model.scheduler.timesteps[-i - 1]
            if i + 1 < len(self.model.scheduler.timesteps):
                next_timestep = self.model.scheduler.timesteps[- i - 2]
            else:
                next_timestep = 999

            alpha_prod_t = self.model.scheduler.alphas_cumprod[timestep]
            alpha_prod_t_next = self.model.scheduler.alphas_cumprod[next_timestep]

            alpha_slice = alpha_prod_t_next / alpha_prod_t

            latent = torch.sqrt(alpha_slice) * latent + torch.sqrt(1 - alpha_slice) * torch.randn_like(latent)
            all_latent.append(latent)
        return all_latent

    def _model_unpatch(self, model):
        # def new_forward_info(self):
        #     def patched_forward(
        #             hidden_states,
        #             encoder_hidden_states=None,
        #             attention_mask=None,
        #             temb=None,
        #     ):
        #         residual = hidden_states

        #         if self.spatial_norm is not None:
        #             hidden_states = self.spatial_norm(hidden_states, temb)

        #         input_ndim = hidden_states.ndim

        #         if input_ndim == 4:
        #             batch_size, channel, height, width = hidden_states.shape
        #             hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        #         batch_size, sequence_length, _ = (
        #             hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        #         )
        #         attention_mask = self.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        #         if self.group_norm is not None:
        #             hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        #         query = self.to_q(hidden_states)

        #         ## Injection
        #         is_self = encoder_hidden_states is None

        #         if encoder_hidden_states is None:
        #             encoder_hidden_states = hidden_states
        #         elif self.norm_cross:
        #             encoder_hidden_states = self.norm_encoder_hidden_states(encoder_hidden_states)

        #         key = self.to_k(encoder_hidden_states)
        #         value = self.to_v(encoder_hidden_states)

        #         query = self.head_to_batch_dim(query)
        #         key = self.head_to_batch_dim(key)
        #         value = self.head_to_batch_dim(value)

        #         attention_probs = self.get_attention_scores(query, key, attention_mask)
        #         hidden_states = torch.bmm(attention_probs, value)
        #         hidden_states = self.batch_to_head_dim(hidden_states)

        #         # linear proj
        #         hidden_states = self.to_out[0](hidden_states)
        #         # dropout
        #         hidden_states = self.to_out[1](hidden_states)

        #         if input_ndim == 4:
        #             hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        #         if self.residual_connection:
        #             hidden_states = hidden_states + residual

        #         hidden_states = hidden_states / self.rescale_output_factor

        #         return hidden_states

        #     return patched_forward
        def new_forward_info(self):
            def patched_forward(
                hidden_states,
                encoder_hidden_states=None,
                attention_mask=None,
                temb=None,
                ip_adapter_masks: Optional[torch.Tensor] = None
            ):
                if encoder_hidden_states is None: #self-attn 
                    residual = hidden_states
                    if self.spatial_norm is not None:
                        hidden_states = self.spatial_norm(hidden_states, temb)

                    input_ndim = hidden_states.ndim
                    if input_ndim == 4:
                        batch_size, channel, height, width = hidden_states.shape
                        hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

                    batch_size, sequence_length, _ = (
                        hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
                    )
                    attention_mask = self.prepare_attention_mask(attention_mask, sequence_length, batch_size)

                    if self.group_norm is not None:
                        hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

                    query = self.to_q(hidden_states) 
                    encoder_hidden_states = hidden_states
                    key = self.to_k(encoder_hidden_states)
                    value = self.to_v(encoder_hidden_states)

                    query = self.head_to_batch_dim(query)
                    key = self.head_to_batch_dim(key)
                    value = self.head_to_batch_dim(value)
                    attention_probs = self.get_attention_scores(query, key, attention_mask)
                       
                    hidden_states = torch.bmm(attention_probs, value)
                    hidden_states = self.batch_to_head_dim(hidden_states)

                    # linear proj
                    hidden_states = self.to_out[0](hidden_states)
                    # dropout
                    hidden_states = self.to_out[1](hidden_states)

                    if input_ndim == 4:
                        hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

                    if self.residual_connection:
                        hidden_states = hidden_states + residual

                    hidden_states = hidden_states / self.rescale_output_factor
                    return hidden_states
                else: # cross_attention 
                    residual = hidden_states
                    if self.spatial_norm is not None:
                        hidden_states = self.spatial_norm(hidden_states, temb)

                    input_ndim = hidden_states.ndim

                    if input_ndim == 4:
                        batch_size, channel, height, width = hidden_states.shape
                        hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

                    batch_size, sequence_length, _ = (
                        hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
                    )

                    if attention_mask is not None:
                        attention_mask = self.prepare_attention_mask(attention_mask, sequence_length, batch_size)
                        attention_mask = attention_mask.view(batch_size, self.heads, -1, attention_mask.shape[-1])

                    if self.group_norm is not None:
                        hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

                    query = self.to_q(hidden_states)
                    if encoder_hidden_states.shape[1] == self.processor.full_emb_size:
                        if encoder_hidden_states is None:
                            encoder_hidden_states = hidden_states
                        else:
                            end_pos = encoder_hidden_states.shape[1] - self.processor.num_tokens
                            encoder_hidden_states, ip_hidden_states = (
                                encoder_hidden_states[:, :end_pos, :],
                                encoder_hidden_states[:, end_pos:, :],
                            )
                            if self.norm_cross:
                                encoder_hidden_states = self.norm_encoder_hidden_states(encoder_hidden_states)

                        key = self.to_k(encoder_hidden_states)
                        value = self.to_v(encoder_hidden_states)

                        inner_dim = key.shape[-1]
                        head_dim = inner_dim // self.heads

                        query = query.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)

                        key = key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
                        value = value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
                        hidden_states = F.scaled_dot_product_attention(
                            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
                        )

                        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.heads * head_dim)
                        hidden_states = hidden_states.to(query.dtype)
                        ip_key = self.processor.to_k_ip(ip_hidden_states.clone())
                        ip_value = self.processor.to_v_ip(ip_hidden_states.clone())
                        ip_key = ip_key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
                        ip_value = ip_value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
                        ip_hidden_states = F.scaled_dot_product_attention(
                            query, ip_key, ip_value, attn_mask=None, dropout_p=0.0, is_causal=False
                        )
                        with torch.no_grad():
                            self.attn_map = query @ ip_key.transpose(-2, -1).softmax(dim=-1)

                        ip_hidden_states = ip_hidden_states.transpose(1, 2).reshape(batch_size, -1, self.heads * head_dim)
                        ip_hidden_states = ip_hidden_states.to(query.dtype)

                        if ip_adapter_masks is not None:    #with ip_adapter mask
                            #print("with mask!!!!!!!!!!!!")
                            if not isinstance(ip_adapter_masks, List):
                                ip_adapter_masks = list(ip_adapter_masks.unsqueeze(1))
   
                            for mask in ip_adapter_masks:
                                mask_downsample = IPAdapterMaskProcessor.downsample(
                                    mask[:, 0, :, :], # now we have mask only for 1 obj
                                    batch_size,
                                    ip_hidden_states.shape[1],
                                    ip_hidden_states.shape[2],
                                )
                    
                            mask_downsample = mask_downsample.to(dtype=query.dtype, device=query.device)
                            hidden_states = hidden_states + self.processor.scale * ip_hidden_states * mask_downsample
                        else:
                            hidden_states = hidden_states + self.processor.scale * ip_hidden_states

                        hidden_states = self.to_out[0](hidden_states)
                        hidden_states = self.to_out[1](hidden_states)

                        if input_ndim == 4:
                            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

                        if self.residual_connection:
                            hidden_states = hidden_states + residual

                        hidden_states = hidden_states / self.rescale_output_factor

                        return hidden_states  
                    else:
                        #without image cross-attention 
                        if encoder_hidden_states is None:
                            encoder_hidden_states = hidden_states
                        else:
                            if self.norm_cross:
                                encoder_hidden_states = self.norm_encoder_hidden_states(encoder_hidden_states)

                        key = self.to_k(encoder_hidden_states)
                        value = self.to_v(encoder_hidden_states)

                        inner_dim = key.shape[-1]
                        head_dim = inner_dim // self.heads

                        query = query.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)

                        key = key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
                        value = value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)

                        hidden_states = F.scaled_dot_product_attention(
                            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
                        )

                        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.heads * head_dim)
                        hidden_states = hidden_states.to(query.dtype)
                        # linear proj
                        hidden_states = self.to_out[0](hidden_states)
                        # dropout
                        hidden_states = self.to_out[1](hidden_states)

                        if input_ndim == 4:
                            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

                        if self.residual_connection:
                            hidden_states = hidden_states + residual

                        hidden_states = hidden_states / self.rescale_output_factor

                        return hidden_states           
            return patched_forward

        def register_attn(module):
            if 'Attention' in module.__class__.__name__:
                module.forward = new_forward_info(module)
            elif hasattr(module, 'children'):
                for module_ in module.children():
                    register_attn(module_)

        def remove_hooks(module):
            if hasattr(module, "_forward_hooks"):
                module._forward_hooks: Dict[int, Callable] = OrderedDict()
            if hasattr(module, 'children'):
                for module_ in module.children():
                    remove_hooks(module_)

        register_attn(model.unet)
        remove_hooks(model.unet)
