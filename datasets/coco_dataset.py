import json
import cv2
import numpy as np

from torch.utils.data import Dataset


class CocoSketchDataset(Dataset):
    def __init__(self, split='train', is_t2i=False, full=False):
        self.data = []
        self.split = split
        with open('./data/COCO/{}/prompt_{}_blip.json'.format(split, split), 'rt') as f: # fill50k, COCO
            for line in f:
                self.data.append(json.loads(line))

        self.is_t2i = is_t2i

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        source = cv2.imread('./data/COCO/{}/sketch/'.format(self.split) + source_filename[7:]) # fill50k, COCO
        target = cv2.imread('./data/COCO/{}/color/'.format(self.split) + source_filename[7:]) # fill50k, COCO


        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)


class CocoSegmDataset(Dataset):
    def __init__(self, split='train2017', is_t2i=False, full=False):
        self.data = []
        self.split = split
        with open('./data/COCO/{}/prompt.json'.format(split), 'rt') as f: # fill50k, COCO
            for line in f:
                self.data.append(json.loads(line))

        self.is_t2i = is_t2i

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        source = cv2.imread('./data/COCO/{}/segm/'.format(self.split) + source_filename[7:]) # fill50k, COCO
        target = cv2.imread('./data/COCO/{}/color/'.format(self.split) + source_filename[7:]) # fill50k, COCO


        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)


class CocoPoseDataset(Dataset):
    def __init__(self, split='train2017', is_t2i=False, full=False):
        self.data = []
        self.split = split
        with open('./data/COCO/{}/prompt.json'.format(split), 'rt') as f: # fill50k, COCO
            for line in f:
                self.data.append(json.loads(line))

        self.is_t2i = is_t2i

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        source = cv2.imread('./data/COCO/{}/pose/'.format(self.split) + source_filename[7:]) # fill50k, COCO
        target = cv2.imread('./data/COCO/{}/color/'.format(self.split) + source_filename[7:]) # fill50k, COCO


        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)



class CocoDepthDataset(Dataset):
    def __init__(self, split='train2017', is_t2i=False, full=False):
        self.data = []
        self.split = split
        if split == 'train':
            with open('./data/COCO/{}/prompt_{}_blip.json'.format(split, split), 'rt') as f: # fill50k, COCO
                for line in f:
                    self.data.append(json.loads(line))
        else:
            if full:
                with open('./data/COCO/{}/prompt_{}_blip_full.json'.format(split, split), 'rt') as f: # fill50k, COCO
                    for line in f:
                        self.data.append(json.loads(line))
            else:
                with open('./data/COCO/{}/prompt_{}_blip.json'.format(split, split), 'rt') as f: # fill50k, COCO
                    for line in f:
                        self.data.append(json.loads(line))

        self.is_t2i = is_t2i

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        source = cv2.imread('./data/COCO/{}/depth/'.format(self.split) + source_filename[7:]) # fill50k, COCO
        target = cv2.imread('./data/COCO/{}/color/'.format(self.split) + source_filename[7:]) # fill50k, COCO


        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0
        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        if self.is_t2i:
            target = np.transpose(target, (2, 0, 1))
            source = np.transpose(source, (2, 0, 1))

        return dict(jpg=target, txt=prompt, hint=source)


class CocoCannyDataset(Dataset):
    def __init__(self, split='train', is_t2i=False, full=False):
        self.data = []
        self.split = split
        if split == 'train':
            with open('./data/COCO/{}/prompt_{}_blip.json'.format(split, split), 'rt') as f: # fill50k, COCO
                for line in f:
                    self.data.append(json.loads(line))
        else:
            if full:
                with open('./data/COCO/{}/prompt_{}_blip_full.json'.format(split, split), 'rt') as f: # fill50k, COCO
                    for line in f:
                        self.data.append(json.loads(line))
            else:
                with open('./data/COCO/{}/prompt_{}_blip.json'.format(split, split), 'rt') as f: # fill50k, COCO
                    for line in f:
                        self.data.append(json.loads(line))

        self.is_t2i = is_t2i

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        source = cv2.imread('./data/COCO/{}/canny/'.format(self.split) + source_filename[7:]) # fill50k, COCO
        target = cv2.imread('./data/COCO/{}/color/'.format(self.split) + source_filename[7:]) # fill50k, COCO


        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0
        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        if self.is_t2i:
            target = np.transpose(target, (2, 0, 1))
            source = np.transpose(source, (2, 0, 1))

        return dict(jpg=target, txt=prompt, hint=source)


class CocoCanny50KDataset(Dataset):
    def __init__(self, split='train', is_t2i=False, full=False):
        self.data = []
        self.split = split

        if split == 'train':
            with open('./data/COCO/{}/prompt_{}_blip_50k.json'.format(split, split), 'rt') as f: # fill50k, COCO
                for line in f:
                    self.data.append(json.loads(line))
        else:
            if full:
                with open('./data/COCO/{}/prompt_{}_blip_full.json'.format(split, split), 'rt') as f: # fill50k, COCO
                    for line in f:
                        self.data.append(json.loads(line))
            else:
                with open('./data/COCO/{}/prompt_{}_blip.json'.format(split, split), 'rt') as f: # fill50k, COCO
                    for line in f:
                        self.data.append(json.loads(line))

        self.is_t2i = is_t2i

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        source = cv2.imread('./data/COCO/{}/canny/'.format(self.split) + source_filename[7:]) # fill50k, COCO
        target = cv2.imread('./data/COCO/{}/color/'.format(self.split) + source_filename[7:]) # fill50k, COCO


        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0
        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        if self.is_t2i:
            target = np.transpose(target, (2, 0, 1))
            source = np.transpose(source, (2, 0, 1))

        return dict(jpg=target, txt=prompt, hint=source)


class CocoCanny20KDataset(Dataset):
    def __init__(self, split='train2017', is_t2i=False, full=False):
        self.data = []
        self.split = split

        if split == 'train':
            with open('./data/COCO/{}/prompt_{}_blip_20k.json'.format(split, split), 'rt') as f: # fill50k, COCO
                for line in f:
                    self.data.append(json.loads(line))
        else:
            if full:
                with open('./data/COCO/{}/prompt_{}_blip_full.json'.format(split, split), 'rt') as f: # fill50k, COCO
                    for line in f:
                        self.data.append(json.loads(line))
            else:
                with open('./data/COCO/{}/prompt_{}_blip.json'.format(split, split), 'rt') as f: # fill50k, COCO
                    for line in f:
                        self.data.append(json.loads(line))

        self.is_t2i = is_t2i

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        source = cv2.imread('./data/COCO/{}/canny/'.format(self.split) + source_filename[7:]) # fill50k, COCO
        target = cv2.imread('./data/COCO/{}/color/'.format(self.split) + source_filename[7:]) # fill50k, COCO


        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0
        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        if self.is_t2i:
            target = np.transpose(target, (2, 0, 1))
            source = np.transpose(source, (2, 0, 1))

        return dict(jpg=target, txt=prompt, hint=source)


class CocoCanny5KDataset(Dataset):
    def __init__(self, split='train2017', is_t2i=False, full=False):
        self.data = []
        self.split = split

        if split == 'train':
            with open('./data/COCO/{}/prompt_{}_blip_5k.json'.format(split, split), 'rt') as f: # fill50k, COCO
                for line in f:
                    self.data.append(json.loads(line))
        else:
            if full:
                with open('./data/COCO/{}/prompt_{}_blip_full.json'.format(split, split), 'rt') as f: # fill50k, COCO
                    for line in f:
                        self.data.append(json.loads(line))
            else:
                with open('./data/COCO/{}/prompt_{}_blip.json'.format(split, split), 'rt') as f: # fill50k, COCO
                    for line in f:
                        self.data.append(json.loads(line))

        self.is_t2i = is_t2i

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        source = cv2.imread('./data/COCO/{}/canny/'.format(self.split) + source_filename[7:]) # fill50k, COCO
        target = cv2.imread('./data/COCO/{}/color/'.format(self.split) + source_filename[7:]) # fill50k, COCO


        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0
        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        if self.is_t2i:
            target = np.transpose(target, (2, 0, 1))
            source = np.transpose(source, (2, 0, 1))

        return dict(jpg=target, txt=prompt, hint=source)


class CocoCanny1KDataset(Dataset):
    def __init__(self, split='train2017', is_t2i=False, full=False):
        self.data = []
        self.split = split

        if split == 'train':
            with open('./data/COCO/{}/prompt_{}_blip_1k.json'.format(split, split), 'rt') as f: # fill50k, COCO
                for line in f:
                    self.data.append(json.loads(line))
        else:
            if full:
                with open('./data/COCO/{}/prompt_{}_blip_full.json'.format(split, split), 'rt') as f: # fill50k, COCO
                    for line in f:
                        self.data.append(json.loads(line))
            else:
                with open('./data/COCO/{}/prompt_{}_blip.json'.format(split, split), 'rt') as f: # fill50k, COCO
                    for line in f:
                        self.data.append(json.loads(line))
            

        self.is_t2i = is_t2i

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        source = cv2.imread('./data/COCO/{}/canny/'.format(self.split) + source_filename[7:]) # fill50k, COCO
        target = cv2.imread('./data/COCO/{}/color/'.format(self.split) + source_filename[7:]) # fill50k, COCO


        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0
        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        if self.is_t2i:
            target = np.transpose(target, (2, 0, 1))
            source = np.transpose(source, (2, 0, 1))

        return dict(jpg=target, txt=prompt, hint=source)