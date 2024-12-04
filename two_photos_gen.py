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
import os

print("all imports are done")

scheduler = DDIMScheduler(
    num_inference_steps=50,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    set_alpha_to_one=False
)

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
base_model_path = "./stable-diffusion-v1-5"
vae_model_path = "./sd-vae-ft-mse"
vae = AutoencoderKL.from_pretrained(vae_model_path)
print("SD + IP-adapter is loading...")

ldm_stable = StableDiffusionPipeline.from_pretrained(
    base_model_path,
    use_safetensors=True,
    scheduler=scheduler,
    vae = vae,
    add_watermarker=False,
).to(device)
ldm_stable.disable_xformers_memory_efficient_attention()
   
config = OmegaConf.load('configs/materialfusion_best.yaml')
image_encoder_path = "models/image_encoder"
ip_ckpt = "models/ip-adapter_sd15.bin"
guidance = GuidanceEditing(ldm_stable, config, image_encoder_path, ip_ckpt, device)
print("guidance object loading is done!")


init_prompts = [
    "A photo of the statue of David", 
    "A photo of the statue of David", 
    ]

edit_prompts = [
    "A photo of the golden statue of David",
    "A photo of the brick statue of David",  
    ]



obj_pathes = ["/home/mdnikolaev/garifullin/gar_zest/example_images/david.jpg", "/home/mdnikolaev/garifullin/gar_zest/example_images/david.jpg"]
objs = ['david', 'david']
ip_image_pathes = ['/home/mdnikolaev/garifullin/gar_zest/demo_assets/material_exemplars/gold_bar.png',
                   '/home/mdnikolaev/garifullin/gar_zest/demo_assets/material_exemplars/kirpich2.jpg']



for j in range(len(obj_pathes)): 
    init_image_path = obj_pathes[j]
    init_image = Image.fromarray(load_512(init_image_path))
    init_image_name = '/home/mdnikolaev/garifullin/MaterialFusion/' + objs[j] +'_init_image.png'
    init_image.save(init_image_name)
    rm_bg = remove(init_image)
    target_mask = rm_bg.convert("RGB").point(lambda x: 0 if x < 1 else 255).convert('L')#.convert('RGB')# Convert mask to grayscale
    #print(target_mask)
    target_mask_name = '/home/mdnikolaev/garifullin/MaterialFusion/' + objs[j] + '_target_mask.png'
    target_mask.save(target_mask_name)
    mask_new = load_image(target_mask_name) # kolhoz, zapisali prochitali
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
    print("edit_prompt: ", edit_prompt)
    scales = [0.1, 0.2, 0.3, 0.4, 0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    for scale in scales:
        res = guidance(init_image, init_prompt, edit_prompt, ip_image, scale = scale, verbose=True, background_mask = 40, cross_attention_kwargs={"ip_adapter_masks": masks})
        
        #print(count, "object editing finished!")
        image_name = 'm_' + objs[j] + str(scale) + str(j) +'.png'
        save_path = '/home/mdnikolaev/garifullin/MaterialFusion/test/' + image_name
        #save_path = '/home/mdnikolaev/garifullin/gar_zest/test_github/' + objs[j] +'/' + image_name
        plt.imsave(save_path, res)


