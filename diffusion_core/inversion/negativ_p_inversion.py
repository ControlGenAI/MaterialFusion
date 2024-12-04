import PIL
import torch

from PIL import Image
from typing import Optional, Union, Tuple, List, Callable, Dict

from .null_inversion import Inversion


class NegativePromptInversion(Inversion):
    def negative_prompt_inversion(self, latents, verbose=False):
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        uncond_embeddings_list = [cond_embeddings.detach()] * self.infer_steps
        return uncond_embeddings_list
    
    def __call__(
        self, 
        image_gt: PIL.Image.Image, 
        prompt: Union[str, torch.Tensor],
        control_image: Optional[PIL.Image.Image] = None,
        verbose: bool = False
    ):
        image_rec, latents, _ = super().__call__(image_gt, prompt, control_image, verbose)
        
        if verbose:
            print("[Negative-Prompt inversion]")
        
        uncond_embeddings = self.negative_prompt_inversion(
            latents, 
            verbose
        )
        return image_rec, latents, uncond_embeddings
