import torch
from diffusers.pipelines import StableDiffusionPipeline
from .utils import ClassRegistry
from .utils.masactrl_utils import MasaCtrlPipeline

diffusion_models_registry = ClassRegistry()


@diffusion_models_registry.add_to_registry("stable-diffusion-v1-4")
def read_v14(scheduler):
    model_id = "CompVis/stable-diffusion-v1-4"
    model = StableDiffusionPipeline.from_pretrained(
        model_id, torch_dtype=torch.float32, scheduler=scheduler
    )
    return model


@diffusion_models_registry.add_to_registry("stable-diffusion-v1-5")
def read_v15(scheduler):
    model_id = "runwayml/stable-diffusion-v1-5"
    model = StableDiffusionPipeline.from_pretrained(
        model_id, torch_dtype=torch.float32, scheduler=scheduler
    )
    return model


@diffusion_models_registry.add_to_registry("stable-diffusion-v1-4-masactrl")
def read_v14_masactrl(scheduler):
    model_id = "CompVis/stable-diffusion-v1-4"
    model = MasaCtrlPipeline.from_pretrained(
        model_id, torch_dtype=torch.float32, scheduler=scheduler
    )
    return model


@diffusion_models_registry.add_to_registry("stable-diffusion-v2-1")
def read_v21(scheduler):
    model_id = "stabilityai/stable-diffusion-2-1"
    model = StableDiffusionPipeline.from_pretrained(
        model_id, torch_dtype=torch.float32, scheduler=scheduler
    )
    return model
