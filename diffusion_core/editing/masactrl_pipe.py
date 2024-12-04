import json
import random
import torch
import numpy as np
import PIL
import gc
from torchvision import transforms
from torchvision.utils import make_grid

import diffusion_core.p2p.utils as ptp_utils


from collections import OrderedDict
from typing import Optional, Union, Tuple, List, Callable, Dict
from tqdm.auto import trange, tqdm

from diffusion_core.guiders.opt_guiders import opt_registry
from diffusion_core.diffusion_utils import latent2image, image2latent
from diffusion_core.p2p.controllers import make_controller, EmptyControl
from diffusion_core.inversion import Inversion, NullInversion, NegativePromptInversion
from diffusion_core.utils.masactrl_utils import (
    AttentionBase, AttentionStore, 
    register_attention_editor_diffusers, 
    MutualSelfAttentionControl,
    MutualSelfAttentionControlProx
)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class MasaCtrlEditPipe:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.verbose = config.verbose
    
    def _convert_image(self, image_pil, device):
        image = np.array(image_pil)
        image = torch.from_numpy(image).unsqueeze(0).float() / 127.5 - 1.
        image = image.to(device).permute(0, 3, 1, 2)
        return image
    
    def to_im(self, torch_image, **kwargs):
        return transforms.ToPILImage()(
            make_grid(torch_image, **kwargs)
        )
    
    def __call__(
        self,
        image_gt: PIL.Image.Image,
        src_prompt: str,
        trg_prompt: str,
        control_image: Optional[PIL.Image.Image] = None,
        verbose: bool = False
    ):
        self._pre_pipeline(verbose)
        s_prompt = src_prompt if self.config.pipe_args.proximal else ''
        self.train(image_gt, s_prompt, trg_prompt)
        result = self.edit()
        self._post_pipeline()
        return result
    
    def _pre_pipeline(self, verbose):
        self.config_verbose = self.verbose
        self.verbose = verbose
        register_attention_editor_diffusers(self.model, AttentionBase())
    
    def _post_pipeline(self):
        register_attention_editor_diffusers(self.model, AttentionBase())
    
    def train(
        self,
        image_gt: PIL.Image.Image,
        src_prompt: str,
        trg_prompt: str
    ):
        seed = 42
        seed_everything(seed)
        source_image = self._convert_image(image_gt, self.model.device)
        self.source_image = source_image
        
        source_prompt = src_prompt
        target_prompt = trg_prompt
        self.prompts = [source_prompt, target_prompt]
        
        # invert the source image
        start_code, latents_list = self.model.invert(
            source_image,
            source_prompt,
            guidance_scale=self.config.pipe_args.inv_scale,
            num_inference_steps=50,
            return_intermediates=True
        )
        
        self.start_code = start_code.expand(len(self.prompts), -1, -1, -1)
        
    @torch.no_grad()
    def edit(self):
        height = width = 512
        
        editor = AttentionBase()
        register_attention_editor_diffusers(self.model, editor)
        image_fixed = self.model(
            [self.prompts[1]],
            latents=self.start_code[-1:],
            num_inference_steps=50,
            guidance_scale=self.config.pipe_args.inference_guidance_scale
        )

        if self.config.pipe_args.proximal:
            editor = MutualSelfAttentionControlProx(
                self.config.pipe_args.step, 
                self.config.pipe_args.layper,
                inject_cond='src',
                inject_uncond='src'
            )
        else:
            editor = MutualSelfAttentionControl(
                self.config.pipe_args.step, 
                self.config.pipe_args.layper
            )
        
        register_attention_editor_diffusers(self.model, editor)
        
        if self.config.pipe_args.proximal:
            prox_args = {
                'prox': 'l0',
                'quantile': 0.6,
                'npi_interp': 1,
                'npi': True,
                'neg_prompt': self.prompts[0],
            }
            gs = [1, self.config.pipe_args.inference_guidance_scale]
        else:
            prox_args = {}
            gs = self.config.pipe_args.inference_guidance_scale
            
        # inference the synthesized image
        image_masactrl = self.model(
            self.prompts,
            latents=self.start_code,
            guidance_scale=gs,
            **prox_args
        )
        
        return np.array(self.to_im(image_masactrl[-1:]))
