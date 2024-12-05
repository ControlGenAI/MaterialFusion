# MaterialFusion: High-Quality, Zero-Shot, and Controllable Material Transfer with Diffusion Models

>Manipulating the material appearance of objects in images is critical for applications like augmented reality, virtual prototyping, and digital content creation. We present MaterialFusion, a novel framework for high-quality material transfer that allows users to adjust the degree of material application, achieving an optimal balance between new material properties and the object's original features. MaterialFusion seamlessly integrates the modified object into the scene by maintaining background consistency and mitigating boundary artifacts. To thoroughly evaluate our approach, we have compiled a dataset of real-world material transfer examples and conducted complex comparative analyses. Through comprehensive quantitative evaluations and user studies, we demonstrate that MaterialFusion significantly outperforms existing methods in terms of quality, user control, and background preservation.
>

![image](docs/teaser.JPG)

## Setup

We ran our code with Python 3.8.5, PyTorch 2.0.1, Diffuser 0.29.1 on NVIDIA V100 GPU with 40GB RAM.

In order to setup the environment, run:
```
conda env create -f material_fusion_env.yaml
```
Conda environment `material_fusion` will be created and you can use it.


## Quickstart

We provide examples of applying our pipeline to real image editing in the [notebook](examples_notebooks/matrial_transfer.ipynb).

## Method Diagram
<p align="center">
  <img src="docs/pipeline.png" alt="Diagram"/>
  <br>
</p>
<p align="center">
  <br>
The overall pipeline of MaterialFusion for material transfer. Starting with DDIM inversion of the target image $x_{init}$ and material exemplar $y_{im}$, the framework combines the IP-Adapter with UNet and employs a guider energy function for precise material transfer. A dual-masking strategy ensures material application only on target regions while preserving background consistency, ultimately generating the edited output $x_{edit}$. The parameter $\lambda$, known as the Material Transfer Force, controls the intensity of the material application, enabling adjustment of the transfer effect according to user preference.
</p>
