# Controlling Text-to-Image Diffusion by Orthogonal Finetuning

## Description

Large text-to-image diffusion models have impressive capabilities in generating photorealistic images from text prompts. How to effectively guide or control these powerful models to perform different downstream tasks becomes an important open problem. To tackle this challenge, we introduce a principled finetuning method -- Orthogonal Finetuning (OFT), for adapting text-to-image diffusion models to downstream tasks. Unlike existing methods, OFT can provably preserve hyperspherical energy which characterizes the pairwise neuron relationship on the unit hypersphere. We find that this property is crucial for preserving the semantic generation ability of text-to-image diffusion models. To improve finetuning stability, we further propose Constrained Orthogonal Finetuning (COFT) which imposes an additional radius constraint to the hypersphere. Specifically, we consider two important finetuning text-to-image tasks: subject-driven generation where the goal is to generate subject-specific images given a few images of a subject and a text prompt, and controllable generation where the goal is to enable the model to take in additional control signals. We empirically show that our OFT framework outperforms existing methods in generation quality and convergence speed.

Stay tuned, more information and code coming soon.

## Table of Contents

1. [Project Title](#project-title)
2. [Description](#description)
3. [Table of Contents](#table-of-contents)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Contributing](#contributing)
7. [License](#license)

## Getting Started

### Downloading Data

1. To download the data required for this project, visit the following link: [https://drive.google.com/drive/folders/1kB3x9KlmklRSeLu74VdH5MEaywvCMVdx?usp=sharing](https://drive.google.com/drive/folders/1kB3x9KlmklRSeLu74VdH5MEaywvCMVdx?usp=sharing)

2. Store the downloaded data in the `data` directory.

After downloading and placing the data, your directory structure should look like this:
```
data
├── ADE20K
│ ├── train
│ │ ├── color
│ │ ├── segm
│ │ └── prompt_train_blip.json
│ └── val
│ │ ├── color
│ │ ├── segm
│ │ └── prompt_val_blip.json
└── COCO
│ ├── train
│ │ ├── color
│ │ ├── depth
│ ...
...
```

### Downloading pre-trained model weights

1. To download the data required for this project, visit the following link: [v1-5-pruned.ckpt](https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main)

2. Store the downloaded model weights in the `models` directory.


## Installation

Follow these steps to set up the project environment:

1. Create the conda environment:
```bash
conda create --name oft python=3.9
conda activate oft
```
2. Install the required packages:
```bash
conda create --name landmark python=3.9
conda activate landmark
pip install torch torchvision torchaudio
pip install open_clip_torch==2.0.2 transformers==4.25.1 omegaconf==2.1.1 pytorch-lightning==1.5.0 gradio==3.16.2 basicsr==1.4.2
pip install matplotlib einops opencv-python pytorch-fid timm fire safetensors pycocotools wandb scikit-image tqdm
pip install torchmetrics[image]
pip install face-alignment==1.3.4
# pip install -r requirements.txt
```

## Usage 

### Controllable Generation

1. Create the model with additional oft parameters:
```bash
python oft-control/tool_add_control_oft.py ./models/v1-5-pruned.ckpt ./models/control_sd15_ini_oft.ckpt
```
2. Train the model (need to specify the control and dataset):
```bash
python oft-control/train.py
```
#### 

### Subject-driven Generation

## Contributing


## Acknowledgements

This project builds upon the work of several other repositories. We would like to express our gratitude to the following projects for their contributions:

- [lora](https://github.com/cloneofsimo/lora): Low-rank Adaptation for Fast Text-to-Image Diffusion Fine-tuning.
- [ControlNet](https://github.com/lllyasviel/ControlNet): Official implementation of Adding Conditional Control to Text-to-Image Diffusion Models.
- [Diffusers](https://github.com/huggingface/diffusers): A library for state-of-the-art pretrained diffusion models.


## License

Add information about your license here. If you're unsure what license to use, GitHub provides a [licensing guide](https://docs.github.com/en/github/creating-cloning-and-archiving-repositories/licensing-a-repository) that might help.

---

### Stay tuned for more updates!
