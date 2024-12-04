import torch
import numpy as np
import torch.nn.functional as nnf
import PIL

from typing import Optional, Union, Tuple, List, Callable, Dict
from tqdm.auto import tqdm, trange
from torch.optim.adam import Adam



from diffusion_core.diffusion_utils import latent2image, image2latent
from diffusion_core.utils.image_utils import load_512
from diffusion_core.custom_forwards.unet_sd import unet_forward
from diffusion_core.schedulers.opt_schedulers import opt_registry


class Inversion:
    def __init__(
        self, model,
        inference_steps, 
        inference_guidance_scale,
        forward_guidance_scale=1,
        verbose=False
    ):
        self.model = model
        self.tokenizer = self.model.tokenizer
        self.scheduler = model.scheduler
        self.scheduler.set_timesteps(inference_steps)
        
        self.prompt = None
        self.context = None
        self.controlnet_cond = None
        
        self.forward_guidance = forward_guidance_scale
        self.backward_guidance = inference_guidance_scale
        self.infer_steps = inference_steps
        self.half_mode = model.unet.dtype == torch.float16
        self.verbose = verbose
        
    @torch.no_grad()
    def init_controlnet_cond(self, control_image):
        if control_image is None:
            return
        
        controlnet_cond = self.model.prepare_image(
            control_image,
            512,
            512,
            1 * 1,
            1,
            self.model.controlnet.device,
            self.model.controlnet.dtype,
        )
        
        self.controlnet_cond = controlnet_cond
    
    def get_noise_pred_single(self, latents, t, context):        
        noise_pred = unet_forward(
            self.model,
            latents,
            t,
            context,
            self.controlnet_cond
        )
        return noise_pred
    
    def get_noise_pred_guided(self, latents, t, guidance_scale, context=None):
        if context is None:
            context = self.context
            
        latents_input = torch.cat([latents] * 2)
        
        if self.controlnet_cond is not None:
            controlnet_cond = torch.cat([self.controlnet_cond] * 2)
            noise_pred = unet_forward(
                self.model,
                latents_input,
                t,
                encoder_hidden_states=context,
                controlnet_cond=controlnet_cond
            )
        else:
            noise_pred = self.model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
            
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
        return noise_pred
    
    @torch.no_grad()
    def init_prompt(self, prompt: Union[str, torch.Tensor]):
        uncond_input = self.model.tokenizer(
            [""], padding="max_length", max_length=self.model.tokenizer.model_max_length,
            return_tensors="pt"
        )
        uncond_embeddings = self.model.text_encoder(uncond_input.input_ids.to(self.model.device))[0]
        if type(prompt) == str:
            text_input = self.model.tokenizer(
                [prompt],
                padding="max_length",
                max_length=self.model.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_embeddings = self.model.text_encoder(text_input.input_ids.to(self.model.device))[0]
        else:
            text_embeddings = prompt
        self.context = torch.cat([uncond_embeddings, text_embeddings])
        self.prompt = prompt
    
    @torch.no_grad()
    def loop(self, latent):
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        all_latent = [latent]
        latent = latent.clone().detach()
        for i in trange(self.infer_steps, desc='Inversion', disable=not self.verbose):
            t = self.scheduler.timesteps[len(self.scheduler.timesteps) - i - 1]

            if not np.allclose(self.forward_guidance, 1.):
                noise_pred = self.get_noise_pred_guided(
                    latent, t, self.forward_guidance, self.context
                )
            else:
                noise_pred = self.get_noise_pred_single(
                    latent, t, cond_embeddings
                )
            
            latent = self.scheduler.step_forward(noise_pred, t, latent).prev_sample

            all_latent.append(latent)
        return all_latent

    @torch.no_grad()
    def inversion(self, image):
        latent = image2latent(image, self.model)
        image_rec = latent2image(latent, self.model)
        latents = self.loop(latent)
        return image_rec, latents
    
    def __call__(
        self, 
        image_gt: PIL.Image.Image, 
        prompt: Union[str, torch.Tensor],
        control_image: Optional[PIL.Image.Image] = None,
        verbose=False
    ):
        self.init_prompt(prompt)
        self.init_controlnet_cond(control_image)
        
        image_gt = np.array(image_gt)
        image_rec, latents = self.inversion(image_gt)
        
        return image_rec, latents, None


class NullInversion(Inversion):
    def null_optimization(self, latents, verbose=False):
        self.scheduler.set_timesteps(self.infer_steps)

        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        uncond_embeddings_list = []
        latent_cur = latents[-1]
        bar = tqdm(total=self.opt_scheduler.max_inner_steps * self.infer_steps)
        
        for i in range(self.infer_steps):
            uncond_embeddings = uncond_embeddings.clone().detach()
            uncond_embeddings.requires_grad = True
            optimizer = Adam([uncond_embeddings], lr=1e-2 * (1. - i / 100.))
            latent_prev = latents[len(latents) - i - 2]
            t = self.scheduler.timesteps[i]
            
            with torch.no_grad():
                noise_pred_cond = self.get_noise_pred_single(latent_cur, t, cond_embeddings)
            
            for j in range(self.opt_scheduler.max_inner_steps):
                noise_pred_uncond = self.get_noise_pred_single(latent_cur, t, uncond_embeddings)
                noise_pred = noise_pred_uncond + self.backward_guidance * (noise_pred_cond - noise_pred_uncond)
                latents_prev_rec = self.scheduler.step_backward(noise_pred, t, latent_cur, first_time=(j == 0), last_time=False).prev_sample
                loss = nnf.mse_loss(latents_prev_rec, latent_prev)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_item = loss.item()
                
                bar.update()      
                if self.opt_scheduler(i, j, loss_item):
                    break
            for j in range(j + 1, self.opt_scheduler.max_inner_steps):
                bar.update()
            uncond_embeddings_list.append(uncond_embeddings[:1].detach())
            
            with torch.no_grad():
                context = torch.cat([uncond_embeddings, cond_embeddings])
                noise_pred = self.get_noise_pred_guided(latent_cur, t, self.backward_guidance, context)
                latent_cur = self.scheduler.step_backward(noise_pred, t, latent_cur, first_time=False, last_time=True).prev_sample
        
        bar.close()
        return uncond_embeddings_list
    
    def __call__(
        self, 
        image_gt: PIL.Image.Image, 
        prompt: Union[str, torch.Tensor],
        opt_scheduler_name: str = 'loss',
        opt_num_inner_steps: int = 10, 
        opt_early_stop_epsilon: float = 1e-5, 
        opt_plateau_prop: float = 1/5,
        control_image: Optional[PIL.Image.Image] = None,
        verbose: bool = False
    ):
        image_rec, latents, _ = super().__call__(image_gt, prompt, control_image, verbose)
        
        if verbose:
            print("Null-text optimization...")
            
        self.opt_scheduler = opt_registry[opt_scheduler_name](
            self.infer_steps, opt_num_inner_steps, 
            opt_early_stop_epsilon, opt_plateau_prop
        )
        
        uncond_embeddings = self.null_optimization(
            latents, 
            verbose
        )
        return image_rec, latents, uncond_embeddings
        