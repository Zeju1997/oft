import torch
from torch.utils.data import Dataset
from torchvision import transforms

from pycocotools.coco import COCO
import os
import numpy as np
from PIL import Image
import json
import cv2


class RescaleToMinusOneToOne:
    def __call__(self, tensor):
        return tensor * 2 - 1


class COCOStuffDataset(Dataset):
    def __init__(self, root_dir, caption_path, set_name, transforms=None):
        self.root_dir = root_dir
        self.transforms = transforms
        self.set_name = set_name
        self.files = []

        # Load captions
        with open(caption_path, 'r', encoding='utf-8') as fp:
            data = json.load(fp)
        data = data['annotations']
        for file in data:
            name = "%012d.png" % file['image_id']
            self.files.append({'name': name, 'sentence': file['caption']})

        # Load annotations
        self.coco = COCO(os.path.join(self.root_dir, "annotations", f"stuff_{self.set_name}.json"))

        # Load all categories
        self.categories = {}
        for category in self.coco.loadCats(self.coco.getCatIds()):
            self.categories[category['id']] = category['name']

        # Load all images
        self.image_ids = self.coco.getImgIds()
        self.images = self.coco.loadImgs(self.image_ids)

    def __getitem__(self, index):
        # Load image
        image_info = self.images[index]
        print('image info', image_info)
        file = self.files[index]
        name = file['name']
        print('name info', name)
        sys.exit()
        image = Image.open(os.path.join(self.root_dir, self.set_name, image_info['file_name']))
        image = np.array(image)

        # Load segmentation map
        segmentation_map = self.load_segmentation_map(image_info['id'])

        # Apply transformations
        if self.transforms is not None:
            image, segmentation_map = self.transforms(image, segmentation_map)

        # Load prompt
        sentence = file['sentence']

        # return image, segmentation_map
        return {'im': image, 'mask': segmentation_map, 'sentence': sentence}

    def __len__(self):
        return len(self.image_ids)

    def load_segmentation_map(self, image_id):
        annotation_ids = self.coco.getAnnIds(imgIds=image_id)
        annotations = self.coco.loadAnns(annotation_ids)

        h, w = self.coco.imgs[image_id]["height"], self.coco.imgs[image_id]["width"]
        segmentation_map = np.zeros((h, w), dtype=np.int32)
        
        for ann in annotations:
            category_id = ann["category_id"]
            mask = self.coco.annToMask(ann)
            segmentation_map[mask == 1] = category_id

        return segmentation_map

class COCOStuffDataset1(Dataset):
    def __init__(self, root_dir, set_name, transforms=None):
        self.root_dir = root_dir
        self.set_name = set_name
        self.transforms = transforms
        self.coco = COCO(os.path.join(root_dir, "annotations/stuff_{}2017.json".format(set_name)))
        self.image_ids = self.coco.getImgIds()

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        # Load image and annotations
        image_info = self.coco.loadImgs(self.image_ids[idx])[0]
        image_path = os.path.join(self.root_dir, self.set_name + "2017", image_info['file_name'])
        image = Image.open(image_path).convert('L')
        ann_ids = self.coco.getAnnIds(imgIds=image_info['id'])
        annotations = self.coco.loadAnns(ann_ids)

        # Create segmentation map
        seg_map = np.zeros_like(np.array(image))
        for ann in annotations:
            cat_id = ann['category_id']
            seg_map[self.coco.annToMask(ann) > 0] = cat_id

        # Apply transforms
        if self.transforms is not None:
            image, seg_map = self.transforms(image, seg_map)

        # Convert to tensor and normalize
        image = torch.from_numpy(np.array(image)).unsqueeze(0).float() / 255.0
        seg_map = torch.from_numpy(seg_map).long()

        return image, seg_map

class Fill50kDataset(Dataset):
    def __init__(self):
        self.data = []
        with open('./training/fill50k/prompt.json', 'rt') as f: # fill50k, COCO
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        # Load source and target images using PIL
        source = Image.open('./training/fill50k/' + source_filename).convert('L')
        target = Image.open('./training/fill50k/' + target_filename).convert('RGB')

        # Convert PIL images to NumPy arrays
        source = np.array(source, dtype=np.float32)
        target = np.array(target, dtype=np.float32)

        target = np.transpose(target, (2, 0, 1))
        source = np.expand_dims(source, axis=0)
        
        # Normalize source images to [0, 1].
        source /= 255.0

        # Normalize target images to [-1, 1].
        target = (target / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)


class CocoSketchDataset(Dataset):
    def __init__(self, split='train2017'):
        self.data = []
        self.split = split
        with open('./training/COCO/{}/prompt.json'.format(split), 'rt') as f: # fill50k, COCO
            for line in f:
                self.data.append(json.loads(line))

        # Define the image transformations
        self.gray_transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor(),
        ])

        self.rgb_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        source_path = os.path.join('./training/COCO/{}/sketch/'.format(self.split), source_filename[7:])
        target_path = os.path.join('./training/COCO/{}/color/'.format(self.split), target_filename[7:])

        source = Image.open(source_path).convert("RGB")
        target = Image.open(target_path).convert("RGB")

        if self.gray_transform:
            source = self.gray_transform(source)

        if self.rgb_transform:
            target = self.rgb_transform(target)

        return dict(jpg=target, txt=prompt, hint=source)


class CocoCannyDataset(Dataset):
    def __init__(self, split='train2017'):
        self.data = []
        self.split = split
        with open('./training/COCO/{}/prompt.json'.format(split), 'rt') as f: # fill50k, COCO
            for line in f:
                self.data.append(json.loads(line))

        # Define the image transformations
        self.gray_transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor(),
        ])

        self.rgb_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        source_path = os.path.join('./training/COCO/{}/canny/'.format(self.split), source_filename[7:])
        target_path = os.path.join('./training/COCO/{}/color/'.format(self.split), target_filename[7:])

        source = Image.open(source_path).convert("RGB")
        target = Image.open(target_path).convert("RGB")

        if self.gray_transform:
            source = self.gray_transform(source)

        if self.rgb_transform:
            target = self.rgb_transform(target)

        return dict(jpg=target, txt=prompt, hint=source)


class CocoDepthDataset(Dataset):
    def __init__(self, split='train2017'):
        self.data = []
        self.split = split
        with open('./training/COCO/{}/prompt.json'.format(split), 'rt') as f: # fill50k, COCO
            for line in f:
                self.data.append(json.loads(line))

        self.rgb_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        source_path = os.path.join('./training/COCO/{}/depth/'.format(self.split), source_filename[7:])
        target_path = os.path.join('./training/COCO/{}/color/'.format(self.split), target_filename[7:])

        source = Image.open(source_path).convert("RGB")
        target = Image.open(target_path).convert("RGB")

        if self.rgb_transform:
            target = self.rgb_transform(target)
            source = self.rgb_transform(source)

        return dict(jpg=target, txt=prompt, hint=source)


class CocoSegmDataset(Dataset):
    def __init__(self, split='train2017'):
        self.data = []
        self.split = split
        with open('./training/COCO/{}/prompt.json'.format(split), 'rt') as f: # fill50k, COCO
            for line in f:
                self.data.append(json.loads(line))

        self.rgb_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        source_path = os.path.join('./training/COCO/{}/segm/'.format(self.split), source_filename[7:])
        target_path = os.path.join('./training/COCO/{}/color/'.format(self.split), target_filename[7:])

        source = Image.open(source_path).convert("RGB")
        target = Image.open(target_path).convert("RGB")

        if self.rgb_transform:
            target = self.rgb_transform(target)
            source = self.rgb_transform(source)

        return dict(jpg=target, txt=prompt, hint=source)

class CocoPoseDataset(Dataset):
    def __init__(self, split='train2017'):
        self.data = []
        self.split = split
        with open('./training/COCO/{}/prompt.json'.format(split), 'rt') as f: # fill50k, COCO
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        source = cv2.imread('./training/COCO/{}/pose/'.format(self.split) + source_filename[7:]) # fill50k, COCO
        target = cv2.imread('./training/COCO/{}/color/'.format(self.split) + source_filename[7:]) # fill50k, COCO


        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)