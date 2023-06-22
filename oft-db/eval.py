#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import hashlib
import logging
import math
import os
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from packaging import version
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig, ViTFeatureExtractor, ViTModel

import lpips
import json
from PIL import Image
import requests
from transformers import AutoProcessor, AutoTokenizer, CLIPModel
import torchvision.transforms.functional as TF
from torch.nn.functional import cosine_similarity
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, ToPILImage


subject_names = [
    "backpack", "backpack_dog", "bear_plushie", "berry_bowl", "can",
    "candle", "cat", "cat2", "clock", "colorful_sneaker",
    "dog", "dog2", "dog3", "dog5", "dog6",
    "dog7", "dog8", "duck_toy", "fancy_boot", "grey_sloth_plushie",
    "monster_toy", "pink_sunglasses", "poop_emoji", "rc_car", "red_cartoon",
    "robot_toy", "shiny_sneaker", "teapot", "vase", "wolf_plushie"
]
'''
subject_names = ["colorful_sneaker"]
'''

class PromptDatasetCLIP(Dataset):
    def __init__(self, image_dir, json_file, tokenizer, processor, epoch=None):
        with open(json_file, 'r') as json_file:
            metadata_dict  = json.load(json_file)
        
        self.image_dir = image_dir
        self.image_lst = []
        self.prompt_lst = []
        for key, value in metadata_dict.items():
            if epoch is not None:
                data_dir = os.path.join(self.image_dir, value['data_dir'], str(epoch))
            else:
                data_dir = os.path.join(self.image_dir, value['data_dir'])
            image_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".png")]
            self.image_lst.extend(image_files[:4])
            class_prompts = [value['instance_prompt']] * len(image_files)
            self.prompt_lst.extend(class_prompts[:4])
        
        print('data_list', len(self.image_lst), len(self.prompt_lst))
        self.tokenizer = tokenizer
        self.processor = processor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __len__(self):
        return len(self.image_lst)

    def __getitem__(self, idx):
        image_path = self.image_lst[idx]
        image = Image.open(image_path)
        prompt = self.prompt_lst[idx]

        extrema = image.getextrema()
        if all(min_val == max_val == 0 for min_val, max_val in extrema):
            return None, None
        else:
            prompt_inputs = self.tokenizer([prompt], padding=True, return_tensors="pt")
            image_inputs = self.processor(images=image, return_tensors="pt")

            return image_inputs, prompt_inputs



class PairwiseImageDatasetCLIP(Dataset):
    def __init__(self, subject, data_dir_A, data_dir_B, processor, epoch):
        self.data_dir_A = data_dir_A
        self.data_dir_B = data_dir_B
        
        self.data_dir_A = os.path.join(self.data_dir_A, subject)
        self.image_files_A = [os.path.join(self.data_dir_A, f) for f in os.listdir(self.data_dir_A) if f.endswith(".jpg")]

        subject = subject + '-'
        self.image_files_B = []
        # Get image files from each subfolder in data A
        for subfolder in os.listdir(data_dir_B):
            if subject in subfolder:
                if epoch is not None:
                    data_dir_B = os.path.join(self.data_dir_B, subfolder, str(epoch))
                else:
                    data_dir_B = os.path.join(self.data_dir_B, subfolder)
                image_files_b = [os.path.join(data_dir_B, f) for f in os.listdir(data_dir_B) if f.endswith(".png")]
                self.image_files_B.extend(image_files_b[:4])

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = processor

    def __len__(self):
        return len(self.image_files_A) * len(self.image_files_B)

    def __getitem__(self, index):
        index_A = index // len(self.image_files_B)
        index_B = index % len(self.image_files_B)
        
        image_A = Image.open(self.image_files_A[index_A]) # .convert("RGB")
        image_B = Image.open(self.image_files_B[index_B]) # .convert("RGB")

        extrema_A = image_A.getextrema()
        extrema_B = image_B.getextrema()
        if all(min_val == max_val == 0 for min_val, max_val in extrema_A) or all(min_val == max_val == 0 for min_val, max_val in extrema_B):
            return None, None
        else:
            inputs_A = self.processor(images=image_A, return_tensors="pt")
            inputs_B = self.processor(images=image_B, return_tensors="pt")

            return inputs_A, inputs_B


class PairwiseImageDatasetDINO(Dataset):
    def __init__(self, subject, data_dir_A, data_dir_B, feature_extractor, epoch):
        self.data_dir_A = data_dir_A
        self.data_dir_B = data_dir_B
        
        self.data_dir_A = os.path.join(self.data_dir_A, subject)
        self.image_files_A = [os.path.join(self.data_dir_A, f) for f in os.listdir(self.data_dir_A) if f.endswith(".jpg")]

        subject = subject + '-'
        self.image_files_B = []
	    # Get image files from each subfolder in data A
        for subfolder in os.listdir(data_dir_B):
            if subject in subfolder:
                if epoch is not None:
                    data_dir_B = os.path.join(self.data_dir_B, subfolder, str(epoch))
                else:
                    data_dir_B = os.path.join(self.data_dir_B, subfolder)
                image_files_b = [os.path.join(data_dir_B, f) for f in os.listdir(data_dir_B) if f.endswith(".png")]        
                self.image_files_B.extend(image_files_b[:4])

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_extractor = feature_extractor

    def __len__(self):
        return len(self.image_files_A) * len(self.image_files_B)

    def __getitem__(self, index):
        index_A = index // len(self.image_files_B)
        index_B = index % len(self.image_files_B)
        
        image_A = Image.open(self.image_files_A[index_A]) # .convert("RGB")
        image_B = Image.open(self.image_files_B[index_B]) # .convert("RGB")

        extrema_A = image_A.getextrema()
        extrema_B = image_B.getextrema()
        if all(min_val == max_val == 0 for min_val, max_val in extrema_A) or all(min_val == max_val == 0 for min_val, max_val in extrema_B):
            return None, None
        else:
            inputs_A = self.feature_extractor(images=image_A, return_tensors="pt")
            inputs_B = self.feature_extractor(images=image_B, return_tensors="pt")

            return inputs_A, inputs_B


class SelfPairwiseImageDatasetCLIP(Dataset):
    def __init__(self, subject, data_dir, processor):
        self.data_dir_A = data_dir
        self.data_dir_B = data_dir
        
        self.data_dir_A = os.path.join(self.data_dir_A, subject)
        self.image_files_A = [os.path.join(self.data_dir_A, f) for f in os.listdir(self.data_dir_A) if f.endswith(".jpg")]

        self.data_dir_B = os.path.join(self.data_dir_B, subject)
        self.image_files_B = [os.path.join(self.data_dir_B, f) for f in os.listdir(self.data_dir_B) if f.endswith(".jpg")]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = processor

    def __len__(self):
        return len(self.image_files_A) * (len(self.image_files_B) - 1)

    def __getitem__(self, index):
        index_A = index // (len(self.image_files_B) - 1)
        index_B = index % (len(self.image_files_B) - 1)

        # Ensure we don't have the same index for A and B
        if index_B >= index_A:
            index_B += 1

        image_A = Image.open(self.image_files_A[index_A]) # .convert("RGB")
        image_B = Image.open(self.image_files_B[index_B]) # .convert("RGB")
        
        inputs_A = self.processor(images=image_A, return_tensors="pt")
        inputs_B = self.processor(images=image_B, return_tensors="pt")

        return inputs_A, inputs_B


class SelfPairwiseImageDatasetDINO(Dataset):
    def __init__(self, subject, data_dir, feature_extractor):
        self.data_dir_A = data_dir
        self.data_dir_B = data_dir
        
        self.data_dir_A = os.path.join(self.data_dir_A, subject)
        self.image_files_A = [os.path.join(self.data_dir_A, f) for f in os.listdir(self.data_dir_A) if f.endswith(".jpg")]

        self.data_dir_B = os.path.join(self.data_dir_B, subject)
        self.image_files_B = [os.path.join(self.data_dir_B, f) for f in os.listdir(self.data_dir_B) if f.endswith(".jpg")]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_extractor = feature_extractor

    def __len__(self):
        return len(self.image_files_A) * (len(self.image_files_B) - 1)

    def __getitem__(self, index):
        index_A = index // (len(self.image_files_B) - 1)
        index_B = index % (len(self.image_files_B) - 1)

        # Ensure we don't have the same index for A and B
        if index_B >= index_A:
            index_B += 1

        image_A = Image.open(self.image_files_A[index_A]) # .convert("RGB")
        image_B = Image.open(self.image_files_B[index_B]) # .convert("RGB")
        
        inputs_A = self.feature_extractor(images=image_A, return_tensors="pt")
        inputs_B = self.feature_extractor(images=image_B, return_tensors="pt")

        return inputs_A, inputs_B


class SelfPairwiseImageDatasetLPIPS(Dataset):
    def __init__(self, subject, data_dir):
        self.data_dir_A = data_dir
        self.data_dir_B = data_dir
        
        self.data_dir_A = os.path.join(self.data_dir_A, subject)
        self.image_files_A = [os.path.join(self.data_dir_A, f) for f in os.listdir(self.data_dir_A) if f.endswith(".jpg")]

        self.data_dir_B = os.path.join(self.data_dir_B, subject)
        self.image_files_B = [os.path.join(self.data_dir_B, f) for f in os.listdir(self.data_dir_B) if f.endswith(".jpg")]

        self.transform = Compose([
            Resize((512, 512)),
            ToTensor(),
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __len__(self):
        return len(self.image_files_A) * (len(self.image_files_B) - 1)

    def __getitem__(self, index):
        index_A = index // (len(self.image_files_B) - 1)
        index_B = index % (len(self.image_files_B) - 1)

        # Ensure we don't have the same index for A and B
        if index_B >= index_A:
            index_B += 1
        
        image_A = Image.open(self.image_files_A[index_A]) # .convert("RGB")
        image_B = Image.open(self.image_files_B[index_B]) # .convert("RGB")

        extrema_A = image_A.getextrema()
        extrema_B = image_B.getextrema()
        if all(min_val == max_val == 0 for min_val, max_val in extrema_A) or all(min_val == max_val == 0 for min_val, max_val in extrema_B):
            return None, None
        else:
            if self.transform:
                image_A = self.transform(image_A)
                image_B = self.transform(image_B)

            return image_A, image_B


class PairwiseImageDatasetLPIPS(Dataset):
    def __init__(self, subject, data_dir_A, data_dir_B, epoch):
        self.data_dir_A = data_dir_A
        self.data_dir_B = data_dir_B
        
        self.data_dir_A = os.path.join(self.data_dir_A, subject)
        self.image_files_A = [os.path.join(self.data_dir_A, f) for f in os.listdir(self.data_dir_A) if f.endswith(".jpg")]

        subject = subject + '-'
        self.image_files_B = []
        # Get image files from each subfolder in data A
        for subfolder in os.listdir(data_dir_B):
            if subject in subfolder:
                if epoch is not None:
                    data_dir_B = os.path.join(self.data_dir_B, subfolder, str(epoch))
                else:
                    data_dir_B = os.path.join(self.data_dir_B, subfolder)
                image_files_b = [os.path.join(data_dir_B, f) for f in os.listdir(data_dir_B) if f.endswith(".png")]
                self.image_files_B.extend(image_files_b[:4])

        self.transform = Compose([
            Resize((512, 512)),
            ToTensor(),
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __len__(self):
        return len(self.image_files_A) * len(self.image_files_B)

    def __getitem__(self, index):
        index_A = index // len(self.image_files_B)
        index_B = index % len(self.image_files_B)
        
        image_A = Image.open(self.image_files_A[index_A]) # .convert("RGB")
        image_B = Image.open(self.image_files_B[index_B]) # .convert("RGB")

        extrema_A = image_A.getextrema()
        extrema_B = image_B.getextrema()
        if all(min_val == max_val == 0 for min_val, max_val in extrema_A) or all(min_val == max_val == 0 for min_val, max_val in extrema_B):
            return None, None
        else:
            if self.transform:
                image_A = self.transform(image_A)
                image_B = self.transform(image_B)

            return image_A, image_B


def clip_text(image_dir, epoch=None):
    criterion = 'clip_text'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    # Get the text features
    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    # Get the image features
    processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")

    dataset = PromptDatasetCLIP(image_dir, 'metadata.json', tokenizer, processor, epoch)
    dataloader = DataLoader(dataset, batch_size=32)

    similarity = []
    for i in tqdm(range(len(dataset))):
        image_inputs, prompt_inputs = dataset[i]
        if image_inputs is not None and prompt_inputs is not None:
            image_inputs['pixel_values'] = image_inputs['pixel_values'].to(device)
            prompt_inputs['input_ids'] = prompt_inputs['input_ids'].to(device)
            prompt_inputs['attention_mask'] = prompt_inputs['attention_mask'].to(device)
            # print(prompt_inputs)
            image_features = model.get_image_features(**image_inputs)
            text_features = model.get_text_features(**prompt_inputs)

            sim = cosine_similarity(image_features, text_features)

            #image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            #text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
            #logit_scale = model.logit_scale.exp()
            #sim = torch.matmul(text_features, image_features.t()) * logit_scale
            similarity.append(sim.item())

    mean_similarity = torch.tensor(similarity).mean().item()
    print(criterion, 'mean_similarity', mean_similarity)

    return mean_similarity, criterion


def clip_image(image_dir, epoch=None):
    criterion = 'clip_image'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    # Get the image features
    processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")

    similarity = []
    for subject in subject_names:
        dataset = PairwiseImageDatasetCLIP(subject, './data', image_dir, processor, epoch)
        # dataset = SelfPairwiseImageDatasetCLIP(subject, './data', processor)

        for i in tqdm(range(len(dataset))):
            inputs_A, inputs_B = dataset[i]
            if inputs_A is not None and inputs_B is not None:
                inputs_A['pixel_values'] = inputs_A['pixel_values'].to(device)
                inputs_B['pixel_values'] = inputs_B['pixel_values'].to(device) 

                image_A_features = model.get_image_features(**inputs_A)
                image_B_features = model.get_image_features(**inputs_B)

                image_A_features = image_A_features / image_A_features.norm(p=2, dim=-1, keepdim=True)
                image_B_features = image_B_features / image_B_features.norm(p=2, dim=-1, keepdim=True)
            
                logit_scale = model.logit_scale.exp()
                sim = torch.matmul(image_A_features, image_B_features.t()) # * logit_scale
                similarity.append(sim.item())

    mean_similarity = torch.tensor(similarity).mean().item()
    print(criterion, 'mean_similarity', mean_similarity)

    return mean_similarity, criterion


def dino(image_dir, epoch=None):
    criterion = 'dino'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ViTModel.from_pretrained('facebook/dino-vits16').to(device)
    feature_extractor = ViTFeatureExtractor.from_pretrained('facebook/dino-vits16')

    similarity = []
    for subject in subject_names:
        dataset = PairwiseImageDatasetDINO(subject, './data', image_dir, feature_extractor, epoch)
        # dataset = SelfPairwiseImageDatasetDINO(subject, './data', feature_extractor)

        for i in tqdm(range(len(dataset))):
            inputs_A, inputs_B = dataset[i]
            if inputs_A is not None and inputs_B is not None:
                inputs_A['pixel_values'] = inputs_A['pixel_values'].to(device)
                inputs_B['pixel_values'] = inputs_B['pixel_values'].to(device) 

                outputs_A = model(**inputs_A)
                image_A_features = outputs_A.last_hidden_state[:, 0, :]

                outputs_B = model(**inputs_B)
                image_B_features = outputs_B.last_hidden_state[:, 0, :]

                image_A_features = image_A_features / image_A_features.norm(p=2, dim=-1, keepdim=True)
                image_B_features = image_B_features / image_B_features.norm(p=2, dim=-1, keepdim=True)

                sim = torch.matmul(image_A_features, image_B_features.t()) # * logit_scale
                similarity.append(sim.item())

    mean_similarity = torch.tensor(similarity).mean().item()
    print(criterion, 'mean_similarity', mean_similarity)

    return mean_similarity, criterion


def lpips_image(image_dir, epoch=None):
    criterion = 'lpips_image'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Set up the LPIPS model (vgg=True uses the VGG-based model from the paper)
    # loss_fn = lpips.LPIPS(net='vgg').to(device)
    loss_fn = lpips.LPIPS(net='alex').to(device)

    similarity = []
    for subject in subject_names:
        dataset = PairwiseImageDatasetLPIPS(subject, './data', image_dir, epoch)
        # dataset = SelfPairwiseImageDatasetLPIPS(subject, './data')
        
        for i in tqdm(range(len(dataset))):
            image_A, image_B = dataset[i]
            if image_A is not None and image_B is not None:
                image_A = image_A.to(device)
                image_B = image_B.to(device)

                # Calculate LPIPS between the two images
                distance = loss_fn(image_A, image_B)

                similarity.append(distance.item())

    mean_similarity = torch.tensor(similarity).mean().item()
    print(criterion, 'LPIPS distance', mean_similarity)

    return mean_similarity, criterion


if __name__ == "__main__":
    image_dir = './log_lora' # './log_db' './log_lora' './log_cot'
    epoch = 8 # None, 2, 4, 6

    sim, criterion = clip_text(image_dir, epoch) # db: 0.1687191128730774 # lora: 0.1668722778558731 # cot: 0.16917628049850464
    # sim, criterion = dino(image_dir, epoch)
    # sim, criterion = clip_image(image_dir, epoch) # db: 18.00 # lora: 18.00 # cot: 18.00
    # sim, criterion = lpips_image(image_dir, epoch) # db: 0.737432599067688 # lora: 18.00 # cot: 0.7569993138313293

    if epoch:
        name = image_dir + '-' + str(epoch)
    else:
        name = image_dir

    filename = "results.txt"  # the name of the file to save the value to
    # Check if file already exists
    file_exists = os.path.isfile(filename)

    # Open the file in append mode if it exists, otherwise create a new file
    with open(filename, "a" if file_exists else "w") as file:
        # If the file exists, add a new line before writing the new data
        if file_exists:
            file.write("\n")
        # Write the name and value as a comma-separated string to the file
        file.write(f"{criterion},{name},{sim}")
