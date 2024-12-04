import torch
import numpy as np
import PIL

from tqdm.auto import tqdm
from torch.optim.adam import Adam
from typing import Optional, Union, List, Tuple

from diffusion_core.diffusion_utils import image2latent, latent2image


class LearnedEmbedding:
    def __init__(self, inversion):
        self.inversion = inversion
        self.emb_opt = None
        self.emb_tgt = None
        self.latent = None
        self.uncond_embeddings = None

    def train(
        self, 
        image_gt: PIL.Image.Image,
        prompt: str,
        learning_rate=5e-3,
        num_iters=1000,
        reg_coeff=1e-3,
        control_image: Optional[PIL.Image.Image] = None,
        num_inner_steps=10, 
        early_stop_epsilon=1e-5,
        verbose=False
    ):
        if verbose:
            print("Learning an embedding...")
        image_gt_numpy = np.array(image_gt)
        emb_opt, emb_tgt = self.learn_emb(image_gt_numpy, prompt, learning_rate, num_iters, reg_coeff)
        self.emb_opt = emb_opt
        self.emb_tgt = emb_tgt

        _, latents, uncond_embeddings = self.inversion(
            image_gt,
            emb_opt,
            control_image,
            num_inner_steps,
            early_stop_epsilon,
            verbose
        )
        self.latent = latents[-1]
        self.uncond_embeddings = uncond_embeddings

    def __call__(
        self,
        alpha_range: Union[List, Tuple, float] = (0, 0.2, 0.4, 0.6, 0.8, 1),
        interpolation_steps_range: Union[List, Tuple, int] = (10, 20, 30, 40, 50)
    ):
        assert self.latent is not None, 'Call "train" method first'
        edited_images = self.generate_with_interpolation(alpha_range, interpolation_steps_range)
        return edited_images

    def learn_emb(self, image, prompt, learning_rate, num_iters, reg_coeff):
        # Get an original embedding, corresponding to the editing prompt (e_tgt)
        text_input = self.inversion.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.inversion.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        emb_tgt = self.inversion.model.text_encoder(text_input.input_ids.to(self.inversion.model.device))[0]

        # Learn an embedding, corresponding to the initial image (e_opt)
        emb_opt = emb_tgt.clone()
        emb_opt = emb_opt.detach()
        emb_opt.requires_grad_()
        emb_tgt = emb_tgt.detach()
        emb_tgt.requires_grad_(False)
        self.change_diffusion_mode(is_train=False)
        
        latents = image2latent(image, self.inversion.model)
        optimizer = Adam([emb_opt], lr=learning_rate)
        criterion = torch.nn.MSELoss()
        
        pbar = tqdm(range(num_iters))
        for i in pbar:
            noise = torch.randn(latents.shape).to(latents.device)
            timesteps = torch.randint(1000, (1,), device=latents.device)
            noisy_latents = self.inversion.model.scheduler.add_noise(latents, noise, timesteps)
            noise_pred = self.inversion.model.unet(noisy_latents, timesteps, emb_opt).sample

            loss = criterion(noise_pred, noise) + reg_coeff * ((emb_opt - emb_tgt) ** 2).sum()
            loss.backward()
            pbar.set_postfix({"loss": loss.item()})
            optimizer.step()
            optimizer.zero_grad()
        emb_opt.requires_grad_(False)
        self.change_diffusion_mode(is_train=True)
        return emb_opt, emb_tgt

    @torch.no_grad()
    def generate_with_interpolation(self, alpha_range, interpolation_steps_range):
        if type(interpolation_steps_range) == int:
            interpolation_steps_range = [interpolation_steps_range]
            tqdm_disable = True
        else:
            tqdm_disable = False
        if type(alpha_range) == float:
            alpha_range = [alpha_range]
        
        all_images = []
        for steps in tqdm(interpolation_steps_range, disable=tqdm_disable):
            all_images.append([])
            for alpha in alpha_range:
                new_emb = alpha * self.emb_tgt + (1-alpha) * self.emb_opt
                image = self.gen_single_image(new_emb, steps)
                all_images[-1].append(image)
        
        if len(all_images) == 1:
            if len(all_images[0]) == 1:
                all_images = all_images[0][0]
            else:
                all_images = all_images[0]

        return all_images
    
    @torch.no_grad()
    def gen_single_image(self, interpolated_emb, interpolation_steps):
        assert interpolation_steps <= self.inversion.infer_steps
        
        latent_cur = self.latent
        cur_text_embedding = self.emb_opt
        for i, t in enumerate(self.inversion.model.scheduler.timesteps[-self.inversion.infer_steps:]):
            if i == self.inversion.infer_steps - interpolation_steps:
                cur_text_embedding = interpolated_emb
            context = torch.cat([self.uncond_embeddings[i].expand(*cur_text_embedding.shape), cur_text_embedding])
            noise_pred = self.inversion.get_noise_pred_guided(latent_cur, t, self.inversion.backward_guidance, context)
            latent_cur = self.inversion.prev_step(noise_pred, t, latent_cur)
        return latent2image(latent_cur, self.inversion.model)
    
    def change_diffusion_mode(self, is_train=False):
        self.inversion.model.vae.requires_grad_(is_train)
        self.inversion.model.unet.requires_grad_(is_train)
        self.inversion.model.text_encoder.requires_grad_(is_train)
        if is_train:
            self.inversion.model.unet.train()
            self.inversion.model.vae.train()
            self.inversion.model.text_encoder.train()
        else:
            self.inversion.model.unet.eval()
            self.inversion.model.vae.eval()
            self.inversion.model.text_encoder.eval()
