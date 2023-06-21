import json
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class CelebHQDataset(Dataset):
    def __init__(self, split='train', is_t2i=False, full=False):
        self.data = []
        self.split = split
        if split == 'train':
            with open('./data/celebhq/prompt_{}_blip.json'.format(split, split), 'rt') as f: # fill50k, COCO
                for line in f:
                    self.data.append(json.loads(line))
        else:
            if full:
                with open('./data/celebhq/prompt_{}_blip_full.json'.format(split, split), 'rt') as f: # fill50k, COCO
                    for line in f:
                        self.data.append(json.loads(line))
            else:
                with open('./data/celebhq/prompt_{}_blip.json'.format(split, split), 'rt') as f: # fill50k, COCO
                    for line in f:
                        self.data.append(json.loads(line))
        # import ipdb; ipdb.set_trace()

        self.is_t2i = is_t2i

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, idx):
        item = self.data[0][idx]

        source_filename = item['image']
        prompt = item['prompt']
        # source = cv2.imread('./training/Celebhq/{}/landmark/'.format(self.split) + source_filename[7:]) # fill50k, COCO
        # target = cv2.imread('./training/Celebhq/{}/color/'.format(self.split) + source_filename[7:]) # fill50k, COCO
        
        # source = cv2.imread('/lustre/fast/fast/zqiu/oft/data/celebhq-text/celeba-hq-landmark2d_cond_512/' + source_filename[:-4] +'.png') # fill50k, COCO
        source = cv2.imread('./data/celebhq-text/celeba-hq-landmark2d_cond_512/' + source_filename[:-4] +'.png') # fill50k, COCO
        #resize the images to 512x512
        # target = cv2.resize(cv2.imread('/lustre/fast/fast/zqiu/OPT-Net/training/celebhq-text/celeba-hq-img/' + source_filename), (512, 512)) # fill50k, COCO
        target = cv2.resize(cv2.imread('./data/celebhq-text/celeba-hq-img/' + source_filename), (512, 512)) # fill50k, COCO

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

      
