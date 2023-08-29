import os
import torch
from tqdm import tqdm
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from oft_utils.unet_2d_condition import UNet2DConditionModel

import json
from torch.utils.data import Dataset, DataLoader


class CocoCaptions(Dataset):
    def __init__(self, captions_file):
        with open(captions_file, 'r') as f:
            data = json.load(f)
        
        self.captions = [item['caption'] for item in data['annotations']]
    
    def __len__(self):
        return len(self.captions)
    
    def __getitem__(self, index):
        return self.captions[index]

class CocoCaptionsBlip(Dataset):
    def __init__(self, json_path):
        with open(json_path, 'r') as f:
            self.data = [json.loads(line) for line in f]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]['prompt']
    

if __name__ == '__main__':
    # captions_file = 'captions_val2017.json' 
    # coco_captions = CocoCaptions(captions_file)

    captions_file = 'prompt_val_blip.json' 
    coco_captions = CocoCaptionsBlip(captions_file)
    output_folder = os.path.join(os.getcwd(), 'results', 'wikiart-30000')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    batch_size = 1
    caption_loader = DataLoader(coco_captions, batch_size=batch_size, shuffle=True)
    num_samples = 8

    model_base = "runwayml/stable-diffusion-v1-5"

    pipe = StableDiffusionPipeline.from_pretrained(
        model_base, torch_dtype=torch.float32,
        safety_checker = None,
        requires_safety_checker = False)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    pipe.unet = UNet2DConditionModel.from_pretrained(
        model_base, subfolder="unet", torch_dtype=torch.float32
    )

    oft_model_path = "./sddata/finetune/oft/wikiart/checkpoint-30000"
    pipe.unet.load_attn_procs(oft_model_path)
    pipe.to("cuda")

    captions_data = []
    for idx, caption in enumerate(caption_loader):
        print('[Sample Idx]', idx)
        caption_entry = {"idx": idx, "caption": caption}
        captions_data.append(caption_entry)

        for i in range(num_samples):
            # use half the weights from the LoRA finetuned model and half the weights from the base model
            # image = pipe("giraffe is eating leaves from the tree", num_inference_steps=25, guidance_scale=7.5, cross_attention_kwargs={"scale": 0.5}).images[0]

            # use the weights from the fully finetuned LoRA model
            image = pipe(caption, num_inference_steps=25, guidance_scale=7.5).images[0]

            image_path = os.path.join(output_folder, "sample_{}_{}.png".format(str(idx), str(i)))
            image.save(image_path)

        if idx > 100:
            break

    json_file = os.path.join(output_folder, 'captions.json')
    # Write to JSON file
    with open(json_file, 'w') as f:
        for caption_entry in captions_data:
            json_line = json.dumps(caption_entry)
            f.write(json_line + '\n')



