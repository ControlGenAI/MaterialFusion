import sys
import os
sys.path.append(os.path.abspath(os.path.join('..')))
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from omegaconf import OmegaConf
from diffusers import StableDiffusionPipeline, AutoencoderKL
from diffusion_core.guiders.guidance_editing import GuidanceEditing
from diffusion_core.utils.image_utils import load_512
from diffusion_core.schedulers.sample_schedulers import DDIMScheduler
from rembg import remove
from diffusers.image_processor import IPAdapterMaskProcessor
from diffusers.utils import load_image


parser = argparse.ArgumentParser()
parser.add_argument("--init_prompt", type=str)
parser.add_argument("--edit_prompt", type=str)
parser.add_argument("--obj_name", type=str)
parser.add_argument("--obj_path", type=str)
parser.add_argument("--material_image_path", type=str)
parser.add_argument("--transfer_force", type=str)
args = parser.parse_args()
scales = [float(x) for x in args.transfer_force.split()]


scheduler = DDIMScheduler(
    num_inference_steps=50,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    set_alpha_to_one=False
)

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
ldm_stable = StableDiffusionPipeline.from_pretrained(
    'runwayml/stable-diffusion-v1-5',
    use_safetensors=True,
    scheduler=scheduler,
    vae = vae,
    add_watermarker=False,
).to(device)
ldm_stable.disable_xformers_memory_efficient_attention()
ldm_stable.disable_xformers_memory_efficient_attention()
config = OmegaConf.load('../configs/materialfusion_colab.yaml')
image_encoder_path = "../IP-Adapter/models/image_encoder"
ip_ckpt = "../IP-Adapter/models/ip-adapter_sd15.bin"
guidance = GuidanceEditing(ldm_stable, config, image_encoder_path, ip_ckpt, device)
print("guidance object loading is done!")

obj_pathes = [args.obj_path]
objs = [args.obj_name]
ip_image_pathes = [args.material_image_path]
init_prompts = [args.init_prompt]
edit_prompts = [args.edit_prompt]

for j in range(len(obj_pathes)):
    init_image_path = obj_pathes[j]
    init_image = Image.fromarray(load_512(init_image_path))
    init_image_name = objs[j] +'_init_image.png'
    init_image.save(init_image_name)
    rm_bg = remove(init_image)
    target_mask = rm_bg.convert("RGB").point(lambda x: 0 if x < 1 else 255).convert('L')
    target_mask_name = objs[j] + '_target_mask.png'
    target_mask.save(target_mask_name)
    mask_new = load_image(target_mask_name)
    output_height = 512
    output_width  = 512
    processor = IPAdapterMaskProcessor()
    masks = processor.preprocess([mask_new], height=output_height, width=output_width)
    masks = [masks.reshape(1, masks.shape[0], masks.shape[2], masks.shape[3])]
    init_prompt = init_prompts[j]
    print("init_prompt:", init_prompt)

    ip_image_path = ip_image_pathes[j]
    ip_image = Image.fromarray(load_512(ip_image_path))
    edit_prompt = edit_prompts[j]
    print("edit_prompt:", edit_prompt)
    for scale in scales:
        res = guidance(init_image, init_prompt, edit_prompt, ip_image, scale = scale, verbose=True, background_mask = 40, cross_attention_kwargs={"ip_adapter_masks": masks})
        image_name = 'm_' + objs[j] + str(scale) +'.png'
        save_path = 'output/' + image_name
        plt.imsave(save_path, res)
        print("Image saved")