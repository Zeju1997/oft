from share import *
import config

import sys
import os
import cv2
import einops
import gradio as gr
import numpy as np
import torch
import random
import json

file_path = os.path.abspath(__file__)
parent_dir = os.path.abspath(os.path.dirname(file_path) + '/..')
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from PIL import Image
from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.midas import MidasDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from cldm.lora import inject_trainable_lora, extract_lora_ups_down, inject_trainable_lora_extended

from cldm.opt_lora import inject_trainable_opt, inject_trainable_opt_conv, inject_trainable_opt_extended, inject_trainable_opt_with_norm

from datasets.utils import return_dataset


def process(input_image, prompt, hint_image, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, low_threshold, high_threshold):
    with torch.no_grad():
        # img = resize_image(HWC3(input_image), image_resolution)
        H, W, C = input_image.shape

        #detected_map = apply_canny(input_image, low_threshold, high_threshold)
        #detected_map = HWC3(detected_map)

        # control = torch.from_numpy(hint_image.copy()).float().cuda() / 255.0
        control = torch.from_numpy(hint_image.copy()).float().cuda()
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        # if config.save_memory:
        #     model.low_vram_shift(is_diffusing=False)

        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        # if config.save_memory:
        #     model.low_vram_shift(is_diffusing=True)

        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)

        # if config.save_memory:
        #     model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)]
    # return [255 - hint_image] + results
    return [input_image] + [hint_image] + results


if __name__ == '__main__':
    # Configs
    img_ID = int(sys.argv[1])

    epoch = 19
    control = 'segm'
    _, dataset, data_name, logger_freq, max_epochs = return_dataset(control, full=True)

    # specify the experiment name
    experiment = './log/image_log_opt_lora_{}_{}_eps_1-3_pe_diff_mlp_r_4_cayley_4gpu'.format(data_name, control)
    
    num_samples = 30
    resume_path = os.path.join(experiment, f'model-epoch={epoch:02d}.ckpt')
    sd_locked = True
    only_mid_control = False
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Result directory
    result_dir = os.path.join(experiment, 'results', str(epoch))
    os.makedirs(result_dir, exist_ok=True)
    source_dir = os.path.join(experiment, 'source', str(epoch))
    os.makedirs(source_dir, exist_ok=True)
    hint_dir = os.path.join(experiment, 'hints', str(epoch))
    os.makedirs(hint_dir, exist_ok=True)

    model = create_model('./models/opt_lora_ldm_v15.yaml').cpu()
    model.model.requires_grad_(False)

    unet_opt_params, train_names = inject_trainable_opt(model.model)
    # unet_opt_params, train_names = inject_trainable_opt_conv(model.model)
    # unet_opt_params, train_names = inject_trainable_opt_extended(model.model)
    # unet_opt_params, train_names = inject_trainable_opt_with_norm(model.model)
    
    model.load_state_dict(load_state_dict(resume_path, location='cuda'))
    model = model.cuda()
    ddim_sampler = DDIMSampler(model)

    pack = range(0, len(dataset), 20)
    formatted_data = {}
    for index in range(1):
        # import ipdb; ipdb.set_trace()
        start_point = pack[img_ID]
        idx = start_point + index

        data = dataset[idx]
        input_image, prompt, hint = data['jpg'], data['txt'], data['hint']
        # input_image, hint = input_image.to(device), hint.to(device)
        
        if not os.path.exists(os.path.join(result_dir, f'result_{idx}_0.png')):
            result_images = process(
                input_image=input_image, 
                prompt=prompt,
                hint_image=hint,
                a_prompt="",
                n_prompt="",
                num_samples=num_samples,
                image_resolution=512,
                ddim_steps=50,
                guess_mode=False,
                strength=1,
                scale=9.0,
                seed=-1,
                eta=0.0,
                low_threshold=100,
                high_threshold=200,
            )

            for i, image in enumerate(result_images):
                if i == 0:
                    image = ((image + 1) * 127.5).clip(0, 255).astype(np.uint8)
                    pil_image = Image.fromarray(image)
                    output_path = os.path.join(source_dir, f'image_{idx}.png')
                    pil_image.save(output_path)
                elif i == 1:
                    image = (image * 255).clip(0, 255).astype(np.uint8)
                    # Convert numpy array to PIL Image
                    pil_image = Image.fromarray(image)
                    # Save PIL Image to file
                    output_path = os.path.join(hint_dir, f'hint_{idx}.png')
                    pil_image.save(output_path)
                else:
                    n = i - 2
                    # Convert numpy array to PIL Image
                    pil_image = Image.fromarray(image)
                    # Save PIL Image to file
                    output_path = os.path.join(result_dir, f'result_{idx}_{n}.png')
                    pil_image.save(output_path)

        # formatted_data[f"item{idx}"] = {
        #     "image_name": f'result_{idx}.png',
        #     "prompt": prompt
        # }

    # with open(os.path.join(experiment, 'results_{}.json'.format(img_ID)), 'w') as f:
    #     json.dump(formatted_data, f)

        
