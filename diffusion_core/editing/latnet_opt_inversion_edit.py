import torch
import numpy as np
import torch.nn.functional as nnf
import PIL

from diffusers import DDIMScheduler
from typing import Optional, Union, Tuple, List, Callable, Dict
from tqdm.auto import tqdm, trange
from torch.optim.adam import Adam

from diffusion_core.diffusion_utils import latent2image, image2latent
from diffusion_core.utils.image_utils import load_512
from diffusion_core.custom_forwards.unet_sd import unet_forward


class LearnLatentsInversionEdit:
    def __init__(self, inversion):
        self.inversion = inversion
        self.latent = None
        self.latents_learned = None
    
    def train(
        self, 
        image_gt: PIL.Image.Image, 
        edit_prompt: Union[str, torch.Tensor],
        control_image: Optional[PIL.Image.Image] = None,
        num_inner_steps: int = 1, 
        early_stop_epsilon: float = 1e-5, 
        latents_opt_lr: float = 2500,
        verbose: bool = False
    ):
        self.inversion.init_prompt(edit_prompt)
        self.inversion.init_controlnet_cond(control_image)

        image_gt = np.array(image_gt)
        image_rec, ddim_latents = self.inversion.ddim_inversion(image_gt)
        self.latent = ddim_latents[-1]
        
        if verbose:
            print("Latents optimization...")
        new_latents_list = self.forward_loop(
            ddim_latents,
            num_inner_steps,
            early_stop_epsilon,
            latents_opt_lr,
            verbose
        )
        self.latents_learned  = new_latents_list
    
    def __call__(
        self, learned_latents_steps: Union[List, Tuple, int] = (0, 10, 20, 30, 40, 50),
    ):
        assert self.latent is not None, 'Call "train" method first'
        if type(learned_latents_steps) == int:
            edited_images = self.gen_image_midstep(learned_latents_steps)
        else:
            edited_images = []
            for step in learned_latents_steps:
                edited_images.append(self.gen_image_midstep(step))
        return edited_images
    
    def forward_loop(self, latents, num_inner_steps, epsilon, lr, verbose=False):
        uncond_embeddings, cond_embeddings = self.inversion.context.chunk(2)
        latent_cur = latents[-1]
        bar = tqdm(total=num_inner_steps * self.inversion.infer_steps)
        new_latents_list = [latent_cur]
        
        for i in range(self.inversion.infer_steps):
            latent_prev = latents[len(latents) - i - 2]
            t = self.inversion.model.scheduler.timesteps[i]
            with torch.no_grad():
                noise_pred_cond = self.inversion.get_noise_pred_single(latent_cur, t, cond_embeddings)
                noise_pred_uncond = self.inversion.get_noise_pred_single(latent_cur, t, uncond_embeddings)
            noise_pred = noise_pred_uncond + self.inversion.backward_guidance * (noise_pred_cond - noise_pred_uncond)
            latents_prev_rec = self.inversion.prev_step(noise_pred, t, latent_cur)

            for j in range(num_inner_steps):
                latents_prev_rec = latents_prev_rec.detach().clone()
                latents_prev_rec.requires_grad = True

                loss = nnf.mse_loss(latents_prev_rec, latent_prev)
                loss.backward()
                grad = latents_prev_rec.grad.detach()
                latents_prev_rec = latents_prev_rec - lr * grad

                loss_item = loss.item()
                bar.update()
                bar.set_postfix({"loss": loss_item})
                if loss_item < epsilon + i * 2e-5:
                    break

            for j in range(j + 1, num_inner_steps):
                bar.update()
                losses.append(loss_item)
            new_latents_list.append(latents_prev_rec.detach())
            
            with torch.no_grad():
                latent_cur = latents_prev_rec
            
        bar.close()
        return new_latents_list
    
    @torch.no_grad()
    def gen_image_midstep(self, step):
        uncond_embeddings, text_embeddings = self.inversion.context.chunk(2)
        latent_cur = self.latent
        for i, t in enumerate(self.inversion.model.scheduler.timesteps[-self.inversion.infer_steps:]):
            if step > i + 1:
                latent_cur = self.latents_learned[i + 1]
                continue
            context = torch.cat([uncond_embeddings.expand(*text_embeddings.shape), text_embeddings])
            noise_pred = self.inversion.get_noise_pred_guided(latent_cur, t, self.inversion.backward_guidance, context)
            latent_cur = self.inversion.prev_step(noise_pred, t, latent_cur)
        return latent2image(latent_cur, self.inversion.model)
