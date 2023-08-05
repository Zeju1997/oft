import os
import torch
from tqdm import tqdm
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

import json
from torch.utils.data import Dataset, DataLoader


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


model_base = "runwayml/stable-diffusion-v1-5"

pipe = StableDiffusionPipeline.from_pretrained(
    model_base,
    torch_dtype=torch.float16,
    safety_checker = None,
    requires_safety_checker = False)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

lora_model_path = "./sddata/finetune/lora/coco/checkpoint-20000"
pipe.unet.load_attn_procs(lora_model_path)
pipe.to("cuda")

captions_file = 'captions_val2017.json' # Replace with the correct path
coco_captions = CocoCaptions(captions_file)
output_folder = os.path.join(os.getcwd(), 'results', 'lora-20000')

batch_size = 1
caption_loader = DataLoader(coco_captions, batch_size=batch_size, shuffle=False)
num_samples = 1

captions_data = []
for idx, caption in enumerate(caption_loader):
    print('[Sample Idx {}/{}]'.format(idx, len(coco_captions)))
    caption_entry = {"idx": idx, "caption": caption}
    captions_data.append(caption_entry)

    # use half the weights from the LoRA finetuned model and half the weights from the base model
    # image = pipe("giraffe is eating leaves from the tree.", num_inference_steps=25, guidance_scale=7.5, cross_attention_kwargs={"scale": 0.5}).images[0]

    # use the weights from the fully finetuned LoRA model
    image = pipe(caption, num_inference_steps=25, guidance_scale=7.5).images[0]

    image_path = os.path.join(output_folder, "sample_{}.png".format(str(idx)))
    image.save(image_path)

    # if idx >= 1999:
    #     break

json_file = os.path.join(output_folder, 'captions.json')
# Write to JSON file
with open(json_file, 'w') as f:
    for caption_entry in captions_data:
        json_line = json.dumps(caption_entry)
        f.write(json_line + '\n')