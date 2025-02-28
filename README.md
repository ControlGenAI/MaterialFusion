# MaterialFusion: High-Quality, Zero-Shot, and Controllable Material Transfer with Diffusion Models

<a href="https://arxiv.org/abs/2502.06606"><img src="https://img.shields.io/badge/arXiv-2502.06606-b31b1b.svg" height=22.5></a>
<a href="https://colab.research.google.com/drive/1xkRLp5vAjv0DnVSP6Vf5xH_WPyczHSyt?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" height=22.5></a>
[![License](https://img.shields.io/github/license/AIRI-Institute/al_toolbox)](./LICENSE)


>Manipulating the material appearance of objects in images is critical for applications like augmented reality, virtual prototyping, and digital content creation. We present MaterialFusion, a novel framework for high-quality material transfer that allows users to adjust the degree of material application, achieving an optimal balance between new material properties and the object's original features. MaterialFusion seamlessly integrates the modified object into the scene by maintaining background consistency and mitigating boundary artifacts. To thoroughly evaluate our approach, we have compiled a dataset of real-world material transfer examples and conducted complex comparative analyses. Through comprehensive quantitative evaluations and user studies, we demonstrate that MaterialFusion significantly outperforms existing methods in terms of quality, user control, and background preservation.
>

![image](docs/teaser.JPG)

## Updates

- [27/02/2025] 🎉🎉🎉 MaterialFusion has been accepted by CVPR 2025.
- [09/02/2025] 🔥🔥🔥 MaterialFusion release.

## Prerequisites

To run our method, please ensure you meet the following hardware and software requirements:
- Operating System: Linux
- GPU: NVIDIA V100 with 40GB RAM
- Python Version: 3.8.5
- PyTorch Version: 2.0.1
- Diffusers Version: 0.29.1

## Setup

* Clone this repo:
```bash
git clone https://github.com/ControlGenAI/MaterialFusion.git
cd MaterialFusion
```

* Download IP-Adapter:
```bash
git lfs install
git clone https://huggingface.co/h94/IP-Adapter
```

* Setup the environment. Conda environment `material_fusion` will be created and you can use it.
```bash
conda env create -f material_fusion_env.yaml
```

## Inference

You can use the `main.py` script to perform inference with the model. This allows you to generate visual outputs based on the provided prompts and parameters.

### Parameters 

- `init_prompt`: A text description of the content in the original image (e.g. "A photo of a car").
- `edit_prompt`: A text description of what you expect in the output image (e.g. "A photo of a golden car"). It is possible to use  $y_{src}$  instead of  $y_{trg}$ . Material transfer is successful regardless of whether the target prompt is present or not(i.e. when  $y_{trg}=y_{src}$ ). However, the presence of the target prompt facilitates easier material transfer.
- `transfer_force`: The strength of the transfer effects applied, specified as a space-separated list of floats (e.g., "0.1 0.5 0.8 1.0").
- `obj_name`: The name of the object, which is used for naming the output images (e.g. "car").
- `obj_path`: The file path to the image of the object (e.g. `./example_images/objects/car.png`).
- `material_image_path`: The path to the material image(e.g. `./example_images/materials/4.jpg`).
- `config_path`: The path to the configuration file used for the model (e.g. `./configs/materialfusion_colab.yaml`). For transferring in Colab, the colab configuration works well; however, if your hardware allows, it's better to use the 'best' configuration. 

### Here’s an example of how to perform inference:
```
init_prompt="A photo of a car"                            
edit_prompt="A photo of a car"                            
transfer_force="0.1 0.5 0.8 1.0"                          
obj_name="car"                                            
obj_path="./example_images/objects/car.png"              
material_image_path="./example_images/materials/4.jpg"   
config_path='./configs/materialfusion_colab.yaml' 

python main.py --init_prompt "$init_prompt" \
--edit_prompt "$edit_prompt" \
--transfer_force "$transfer_force" \
--obj_name "$obj_name" \
--obj_path "$obj_path" \
--material_image_path "$material_image_path" \
--config "$config_path"
```

## Quickstart

We provide examples of applying our pipeline to real image editing in the [notebook](example_notebooks/material_transfer.ipynb).

## Method Diagram
<p align="center">
  <img src="docs/pipeline.png" alt="Diagram"/>
  <br>
</p>
<p align="center">
  <br>
The overall pipeline of MaterialFusion for material transfer. Starting with DDIM inversion of the target image $x_{init}$ and material exemplar $y_{im}$, the framework combines the IP-Adapter with UNet and employs a guider energy function for precise material transfer. A dual-masking strategy ensures material application only on target regions while preserving background consistency, ultimately generating the edited output $x_{edit}$. The parameter $\lambda$, known as the Material Transfer Force, controls the intensity of the material application, enabling adjustment of the transfer effect according to user preference.
</p>

## References & Acknowledgments

The repository was started from [Guide-and-Rescale](https://github.com/AIRI-Institute/Guide-and-Rescale).

## Citation

If you utilize this code in your research, kindly cite our paper:
```
@article{garifullin2025materialfusion,
  title={MaterialFusion: High-Quality, Zero-Shot, and Controllable Material Transfer with Diffusion Models},
  author={Kamil Garifullin and Maxim Nikolaev and Andrey Kuznetsov and Aibek Alanov},
  journal={arXiv preprint arXiv:2502.06606},
 year={2025}
}
```


