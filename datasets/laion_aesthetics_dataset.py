import os
import json
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image

from tqdm import tqdm

class LaionAestheticsRawDataset(Dataset):
    def __init__(self, root_dir, chunks=None, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.image_files = []
        self.captions = []
        for subdir in chunks:
            subdir_path = os.path.join(self.root_dir, subdir)
            if os.path.isdir(subdir_path):
                caption_files = sorted([os.path.join(subdir_path, f) for f in os.listdir(subdir_path) if f.endswith('.txt')])
                image_files = sorted([os.path.join(subdir_path, f) for f in os.listdir(subdir_path) if f.endswith('.jpg')])
                assert len(caption_files) == len(image_files), 'Number of captions and images does not match'
                self.image_files.extend(image_files)
                self.captions.extend(caption_files)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        with open(self.captions[idx], 'r') as file:
            caption = file.read()

        return image, caption


class LaionAestheticsRawHumanDataset(Dataset):
    def __init__(self, root_dir, chunks=None, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.image_files = []
        self.captions = []
        for subdir in chunks:
            subdir_path = os.path.join(self.root_dir, subdir)
            if os.path.isdir(subdir_path):
                caption_files = sorted([os.path.join(subdir_path, f) for f in os.listdir(subdir_path) if f.endswith('.txt')])
                image_files = sorted([os.path.join(subdir_path, f) for f in os.listdir(subdir_path) if f.endswith('.jpg')])
                assert len(caption_files) == len(image_files), 'Number of captions and images does not match'
                self.image_files.extend(image_files)
                self.captions.extend(caption_files)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        with open(self.captions[idx], 'r') as file:
            caption = file.read()

        return image, caption


class LaionAestheticsDataset(Dataset):
    def __init__(self, split='train', transform=None):
        self.data = []
        self.split = split
        with open('../data/LAION/prompt_{}.json'.format(split), 'rt') as f: # fill50k, COCO
            for line in f:
                self.data.append(json.loads(line))
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        # source = cv2.imread('../data/LAION/{}/'.format(self.split) + source_filename[7:]) # fill50k, COCO
        image_path = os.path.join('../data/LAION/{}/'.format(self.split), target_filename[7:]) # fill50k, COCO

        # Do not forget that OpenCV read images in BGR order.
        # source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        # target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        # source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        # target = (target.astype(np.float32) / 127.5) - 1.0

        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return dict(img=image, txt=prompt)


if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((512, 512)),  # resize the image to (512, 512)
        transforms.ToTensor()  # convert the image to a tensor
    ])

    # val dataset [00041, ..., 00043]
    image_dir = './data/LAION/val/'
    prompt_path = './data/LAION/prompt_val.json'
    captions = []
    chunks = [f"{i:05d}" for i in range(41, 44)]
    dataset = LaionAestheticsRawDataset(root_dir='./data/laion-aesthetic/laion-aesthetic-data/', chunks=chunks, transform=transform)
    for i in tqdm(range(len(dataset))):
        data = dataset[i]
        image, caption = data
        image_path = os.path.join(image_dir, '{}.png'.format(i))
        captions.append(caption)
        save_image(image, image_path)

    with open(prompt_path, 'w') as f:
        for idx, caption in enumerate(captions):
            line_data = {
                'source': 'source/{}.png'.format(idx), 
                'target': 'target/{}.png'.format(idx), 
                'prompt': caption
            }
            json.dump(line_data, f)
            f.write('\n')

    # train dataset [00000, ..., 00040]
    image_dir = './data/LAION/train/'
    prompt_path = './data/LAION/prompt_train.json'
    captions = []
    chunks = [f"{i:05d}" for i in range(0, 41)]
    dataset = LaionAestheticsDataset(root_dir='./data/laion-aesthetic/laion-aesthetic-data/', chunks=chunks, transform=transform)
    for i in tqdm(range(len(dataset))):
        data = dataset[i]
        image, caption = data
        image_path = os.path.join(image_dir, '{}.png'.format(i))
        captions.append(caption)
        save_image(image, image_path)

    with open(prompt_path, 'w') as f:
        for idx, caption in enumerate(captions):
            line_data = {
                'source': 'source/{}.png'.format(idx), 
                'target': 'target/{}.png'.format(idx), 
                'prompt': caption
            }
            json.dump(line_data, f)
            f.write('\n')