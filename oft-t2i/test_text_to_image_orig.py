import os
import torch
from tqdm import tqdm
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

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
    
    output_folder = os.path.join(os.getcwd(), 'results', 'sd-orig')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    batch_size = 1
    caption_loader = DataLoader(coco_captions, batch_size=batch_size, shuffle=False)
    num_samples = 1

    model_path = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_path, 
        torch_dtype=torch.float16,
        safety_checker = None,
        requires_safety_checker = False)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to("cuda")

    for index in range(5000):
        print(index)
        caption = coco_captions[index]

        image = pipe(prompt=caption, num_inference_steps=25, guidance_scale=7.5).images[0]

        image_path = os.path.join(output_folder, "sample_{}.png".format(str(index)))
        image.save(image_path)

        # if idx >= 1999:
        #     break


