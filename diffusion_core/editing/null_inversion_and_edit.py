import torch
import torch.nn.functional as nnf
import numpy as np
import PIL

from tqdm.auto import tqdm, trange
from torch.optim.adam import Adam
from typing import Optional, Union, List, Tuple

from diffusion_core.diffusion_utils import latent2image


class NullInversionEdit:
    def __init__(self, inversion):
        self.inversion = inversion
        self.latent = None
        self.uncond_embeddings_trained = None
    
    def train(
        self, 
        image_gt: PIL.Image.Image, 
        edit_prompt: Union[str, torch.Tensor],
        control_image: Optional[PIL.Image.Image] = None,
        num_inner_steps=10,
        early_stop_epsilon=1e-5,
        verbose=False
    ):
        self.inversion.init_prompt(edit_prompt)
        self.inversion.init_controlnet_cond(control_image)

        image_gt = np.array(image_gt)
        image_rec, ddim_latents = self.inversion.ddim_inversion(image_gt)
        self.latent = ddim_latents[-1]
        
        if verbose:
            print("Null-text optimization...")
        uncond_embeddings = self.inversion.null_optimization(ddim_latents, num_inner_steps, early_stop_epsilon, verbose)
        self.uncond_embeddings_trained = uncond_embeddings
    
    def __call__(
        self, learned_uncond_steps: Union[List, Tuple, int] = (0, 5, 15, 20, 25, 30)
    ):
        assert self.latent is not None, 'Call "train" method first'

        if type(learned_uncond_steps) == int:
            edited_images = self.gen_image_midstep(learned_uncond_steps)
        else:
            edited_images = []
            for step in learned_uncond_steps:
                edited_images.append(self.gen_image_midstep(step))
        
        return edited_images
    
    @torch.no_grad()
    def gen_image_midstep(self, step):
        _, text_embeddings = self.inversion.context.chunk(2)

        empty_input = self.inversion.model.tokenizer(
            [""], padding="max_length", max_length=self.inversion.model.tokenizer.model_max_length, return_tensors="pt"
        )
        empty_embedding = self.inversion.model.text_encoder(empty_input.input_ids.to(self.inversion.model.device))[0].detach()

        uncond_embeddings = self.uncond_embeddings_trained[:step] + [empty_embedding] * (self.inversion.infer_steps - step)        

        latent_cur = self.latent

        for i, t in enumerate(self.inversion.model.scheduler.timesteps[-self.inversion.infer_steps:]):
            context = torch.cat([uncond_embeddings[i].expand(*text_embeddings.shape), text_embeddings])
            noise_pred = self.inversion.get_noise_pred_guided(latent_cur, t, self.inversion.backward_guidance, context)
            latent_cur = self.inversion.prev_step(noise_pred, t, latent_cur)
        
        return latent2image(latent_cur, self.inversion.model)
