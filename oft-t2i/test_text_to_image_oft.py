import os
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from oft_utils.unet_2d_condition import UNet2DConditionModel

import json
from torch.utils.data import Dataset, DataLoader
import argparse

class CocoCaptions(Dataset):
    def __init__(self, captions_file):
        with open(captions_file, 'r') as f:
            data = json.load(f)
        
        captions = [item['caption'] for item in data['annotations']]
        image_ids = [item['image_id'] for item in data['annotations']]

        # Pairing image_ids and captions and sorting by image_id
        sorted_data = sorted(zip(image_ids, captions), key=lambda x: x[0])

        # Keep only the first caption for each unique image_id
        unique_data = {}
        for image_id, caption in sorted_data:
            if image_id not in unique_data:
                unique_data[image_id] = caption

        self.image_ids, self.captions = zip(*unique_data.items())
    
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_ID', type=int, help='Number of samples')
    args = parser.parse_args()

    # captions_file = 'captions_val2017.json' 
    # coco_captions = CocoCaptions(captions_file)

    captions_file = 'prompt_val_blip.json' 
    coco_captions = CocoCaptionsBlip(captions_file)
    output_folder = os.path.join(os.getcwd(), 'results', 'oft-20000_coco_1e-05')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    model_base = "runwayml/stable-diffusion-v1-5"

    pipe = StableDiffusionPipeline.from_pretrained(
        model_base, 
        torch_dtype=torch.float16,
        safety_checker = None,
        requires_safety_checker = False)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    pipe.unet = UNet2DConditionModel.from_pretrained(
        model_base, subfolder="unet", torch_dtype=torch.float16
    )

    oft_model_path = "./sddata/finetune/oft/coco_1e-05/checkpoint-20000"
    pipe.unet.load_attn_procs(oft_model_path)
    pipe.to("cuda")

    batch_size = 1
    caption_loader = DataLoader(coco_captions, batch_size=batch_size, shuffle=False)
    num_samples = 1

    pack = range(0, 5000, 100)
    for index in range(100):
        start_point = pack[args.img_ID]
        idx = start_point + index
        caption = coco_captions[idx]

        # use half the weights from the LoRA finetuned model and half the weights from the base model
        # image = pipe("giraffe is eating leaves from the tree.", num_inference_steps=25, guidance_scale=7.5, cross_attention_kwargs={"scale": 0.5}).images[0]

        # use the weights from the fully finetuned LoRA model
        image = pipe(caption, num_inference_steps=25, guidance_scale=7.5).images[0]

        image_path = os.path.join(output_folder, "sample_{}.png".format(str(idx)))
        image.save(image_path)