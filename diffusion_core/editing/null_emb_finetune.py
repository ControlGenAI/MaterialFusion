import torch
import torch.nn.functional as nnf
import numpy as np
import PIL

from tqdm.auto import tqdm
from torch.optim.adam import Adam
from typing import Optional, Union, List, Tuple

from diffusion_core.inversion.null_inversion import NullInversion
from diffusion_core.diffusion_utils import latent2image


class NullEmbFinetune(NullInversion):
    def __init__(self, model, inference_steps, inference_guidance_scale):
        super().__init__(model, inference_steps, inference_guidance_scale)
        self.latent = None
        self.uncond_embeddings_init = None
        self.uncond_embeddings_finetuned = None
        
    def train(
        self, 
        image_gt: PIL.Image.Image, 
        prompt: Union[str, torch.Tensor],
        latents: List[torch.Tensor],
        uncond_embeddings_init: Optional[List[torch.Tensor]] = None,
        control_image: Optional[PIL.Image.Image] = None,
        num_inner_steps=30, 
        early_stop_epsilon=1e-5, 
        verbose=False
    ):
        self.init_prompt(prompt)
        self.init_controlnet_cond(control_image)
        self.uncond_embeddings_init = uncond_embeddings_init
        self.latent = latents[-1] 
        image_gt = np.array(image_gt)
        
        if verbose:
            print("Null-text finetuning...")
        uncond_embeddings = self.null_optimization(
            latents,
            num_inner_steps,
            early_stop_epsilon,
            verbose
        )
        self.uncond_embeddings_finetuned = uncond_embeddings


    def __call__(self, gen_steps: Union[List, Tuple, int] = (0, 10, 15, 20, 25, 30)):
        assert self.latent is not None, 'Call "train" method first'

        if type(gen_steps) != int:
            edited_images = []
            for step in tqdm(gen_steps):
                edited_images.append(self.gen_image_midstep(step))
        else:
            edited_images = self.gen_image_midstep(gen_steps)
        return edited_images


    def null_optimization(self, latents, num_inner_steps, epsilon, verbose=False):
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        uncond_embeddings_list = []
        latent_cur = latents[-1]
        bar = tqdm(total=num_inner_steps * self.infer_steps)
        
        for i in range(self.infer_steps):
            # use init for null-text embedding if provided
            if self.uncond_embeddings_init is None:
                uncond_embeddings = uncond_embeddings.clone().detach()
            else:
                uncond_embeddings = self.uncond_embeddings_init[i]
            
            uncond_embeddings.requires_grad = True
            optimizer = Adam([uncond_embeddings], lr=1e-2 * (1. - i / 100.))
            latent_prev = latents[len(latents) - i - 2]
            t = self.model.scheduler.timesteps[i]
            
            with torch.no_grad():
                noise_pred_cond = self.get_noise_pred_single(latent_cur, t, cond_embeddings)
            
            for j in range(num_inner_steps):
                noise_pred_uncond = self.get_noise_pred_single(latent_cur, t, uncond_embeddings)
                noise_pred = noise_pred_uncond + self.backward_guidance * (noise_pred_cond - noise_pred_uncond)
                latents_prev_rec = self.prev_step(noise_pred, t, latent_cur)
                loss = nnf.mse_loss(latents_prev_rec, latent_prev)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_item = loss.item()
                
                bar.update()                
                if loss_item < epsilon + i * 2e-5:
                    break
            for j in range(j + 1, num_inner_steps):
                bar.update()
            uncond_embeddings_list.append(uncond_embeddings[:1].detach())
            
            with torch.no_grad():
                context = torch.cat([uncond_embeddings, cond_embeddings])
                noise_pred = self.get_noise_pred_guided(latent_cur, t, self.backward_guidance, context)
                latent_cur = self.prev_step(noise_pred, t, latent_cur)
        
        bar.close()
        return uncond_embeddings_list

    @torch.no_grad()
    def gen_image_midstep(self, step):
        _, text_embeddings = self.context.chunk(2)

        # TODO:
        # use self.uncond_embeddings_init values for null-text embeddings instead
        empty_input = self.model.tokenizer(
            [""], padding="max_length", max_length=self.tokenizer.model_max_length, return_tensors="pt"
        )
        empty_embedding = self.model.text_encoder(empty_input.input_ids.to(self.model.device))[0].detach()

        uncond_embeddings = self.uncond_embeddings_finetuned[:step] + [empty_embedding] * (self.infer_steps - step)
        latent_cur = self.latent
        for i, t in enumerate(self.model.scheduler.timesteps[-self.infer_steps:]):
            context = torch.cat([uncond_embeddings[i].expand(*text_embeddings.shape), text_embeddings])
            noise_pred = self.get_noise_pred_guided(latent_cur, t, self.backward_guidance, context)
            latent_cur = self.prev_step(noise_pred, t, latent_cur)
        return latent2image(latent_cur, self.model)