import torch
from diffusers.pipelines import StableDiffusionPipeline
from diffusion_core.schedulers import DDIMScheduler
from .utils import ClassRegistry
    

diffusion_schedulers_registry = ClassRegistry()


@diffusion_schedulers_registry.add_to_registry("ddim_50_eps")
def get_ddim_50_e():
    scheduler = DDIMScheduler(
        beta_start=0.00085, 
        beta_end=0.012, 
        beta_schedule="scaled_linear", 
        set_alpha_to_one=False,
        num_inference_steps=50,
        prediction_type='epsilon'
    )
    return scheduler


@diffusion_schedulers_registry.add_to_registry("ddim_50_v")
def get_ddim_50_v():
    scheduler = DDIMScheduler(
        beta_start=0.00085, 
        beta_end=0.012, 
        beta_schedule="scaled_linear", 
        set_alpha_to_one=False,
        num_inference_steps=50,
        prediction_type='v_prediction'
    )
    return scheduler


@diffusion_schedulers_registry.add_to_registry("ddim_200_v")
def get_ddim_200_v():
    scheduler = DDIMScheduler(
        beta_start=0.00085, 
        beta_end=0.012, 
        beta_schedule="scaled_linear", 
        set_alpha_to_one=False,
        num_inference_steps=200,
        prediction_type='v_prediction'
    )
    return scheduler
