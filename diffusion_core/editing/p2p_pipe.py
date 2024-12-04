import json
import torch
import numpy as np
import PIL
import gc

import diffusion_core.p2p.utils as ptp_utils

from collections import OrderedDict
from typing import Optional, Union, Tuple, List, Callable, Dict
from tqdm.auto import trange, tqdm

from diffusion_core.guiders.opt_guiders import opt_registry
from diffusion_core.diffusion_utils import latent2image, image2latent
from diffusion_core.p2p.controllers import make_controller, EmptyControl
from diffusion_core.inversion import Inversion, NullInversion, NegativePromptInversion


class P2PPipeBase:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.verbose = config.verbose
        self._setup_inversion_engine()
    
    def __call__(
        self,
        image_gt: PIL.Image.Image,
        src_prompt: str,
        trg_prompt: str,
        control_image: Optional[PIL.Image.Image] = None,
        verbose: bool = False
    ):
        self._pre_pipeline(verbose)
        self.train(image_gt, src_prompt, trg_prompt)
        result = self.edit()
        self._post_pipeline()
        return result
    
    def _setup_inversion_engine(self):
        if self.config.inversion_type == 'ntinv':
            self.inversion_engine = NullInversion(
                self.model,
                self.model.scheduler.num_inference_steps, 
                self.config.pipe_args.inference_guidance_scale,
                forward_guidance_scale=1,
                verbose=self.config.verbose
            )
        elif self.config.inversion_type == 'npinv':
            self.inversion_engine = NegativePromptInversion(
                self.model,
                self.model.scheduler.num_inference_steps, 
                self.config.pipe_args.inference_guidance_scale,
                forward_guidance_scale=1,
                verbose=self.config.verbose
            )
        else:
            raise ValueError('Incorrect InversionType')
    
    def _pre_pipeline(self, verbose):
        self.config_verbose = self.verbose
        self.verbose = verbose
    
    def _post_pipeline(self):
        self.vernose = self.config_verbose
        delattr(self, 'config_verbose')
        ptp_utils.register_attention_control(self.model, EmptyControl())
    
    def train(
        self,
        image_gt: PIL.Image.Image,
        src_prompt: str,
        trg_prompt: str
    ):
        self.src_prompt, self.trg_prompt = src_prompt, trg_prompt
        
        image_rec, self.lats, self.uncond_embeddings = self.inversion_engine(
            image_gt,
            src_prompt,
            verbose=self.verbose
        )
        
        self.start_latent = self.lats[-1]
        
        is_replace_controller, eq_params = self.__preprocess_prompts(src_prompt, trg_prompt)
        blend_word=None
        
        self.controller = make_controller(
            self.model, 
            [src_prompt, trg_prompt], 
            is_replace_controller,
            self.config.pipe_args.cross_replace_steps,
            self.config.pipe_args.self_replace_steps,
            blend_word,
            eq_params
        )
        ptp_utils.register_attention_control(self.model, self.controller)
        
    def __preprocess_prompts(self, p1, p2):
        tokens1, tokens2 = p1.split(' '), p2.split(' ')

        is_replace_controller = len(tokens1) == len(tokens2)

        matched = [t2 in tokens1 for t2 in tokens2]
        unique_tokens = [tokens2[i] for i, flag in enumerate(matched) if not flag]
        eq_params_ = {
            "words": tuple(unique_tokens),
            "values": tuple([2. for _ in unique_tokens])
        }

        return is_replace_controller, eq_params_
    
    @torch.no_grad()
    def edit(self):
        height = width = 512
        
        text_input = self.model.tokenizer(
            [self.src_prompt, self.trg_prompt],
            padding="max_length",
            max_length=self.model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.model.text_encoder(text_input.input_ids.to(self.model.device))[0]
        max_length = text_input.input_ids.shape[-1]

        latent, latents = ptp_utils.init_latent(
            self.start_latent,
            self.model,
            height=512,
            width=512,
            generator=None,
            batch_size=2
        )
        
        self.model.scheduler.set_timesteps(self.model.scheduler.num_inference_steps)
        
        for i, timestep in tqdm(
            enumerate(self.model.scheduler.timesteps),
            total=self.model.scheduler.num_inference_steps,
            desc='Editing',
            disable=not self.verbose
        ):            
            
            context = torch.cat([self.uncond_embeddings[i].expand(*text_embeddings.shape), text_embeddings])
            
            if self.config.pipe_args.proximal:
                proximal_kwargs = {
                    'prox': 'l0',
                    'quantile': 0.7,
                    'recon_lr': 1,
                    'recon_t': 400,
                    'inversion_guidance': True,
                    'dilate_mask': 1
                }
            else:
                proximal_kwargs = {
                    'prox': None,
                    'recon_lr': 0,
                    'recon_t': 1000
                }
            
            latents = ptp_utils.diffusion_step(
                self.model, 
                self.controller, 
                latents, 
                context, 
                timestep, 
                self.config.pipe_args.inference_guidance_scale, 
                low_resource=False,
                inference_stage=True, 
                x_stars=self.lats, 
                i=i,
                **proximal_kwargs
            )
        
        image = latent2image(latents, self.model)
        return image[1]