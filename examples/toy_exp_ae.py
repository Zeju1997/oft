import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
from torch.utils.data import Dataset

import torchvision.transforms as transforms
from torchvision.utils import save_image
import torchvision.datasets as dsets
from torchvision.transforms.functional import to_tensor, to_pil_image

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Check the dataset
# from datasets import load_dataset


class ImageDataset(Dataset):
    def __init__(self, dir_path, transform=None):
        self.dir_path = dir_path
        self.transform = transform

        # List all the files in the directory
        self.image_files = sorted(os.listdir(dir_path), key=lambda x: int(x.split('_')[1].split('.')[0]))
        # self.image_files = os.listdir(dir_path)
        self.image_files = self.image_files[:1000]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Construct the full path of the image
        image_path = os.path.join(self.dir_path, self.image_files[idx])

        # Open the image file
        image = Image.open(image_path).convert('RGB')

        # Apply the transform to the image
        if self.transform:
            image = self.transform(image)

        return image


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        
        # Encoder layers
        self.encode_conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.encode_conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.encode_conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)

        # Decoder layers
        self.decode_convT1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.decode_convT2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.decode_convT3 = nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1)

    def _get_filter_norm(self, filt):
        eps = 1e-4
        return torch.sqrt(torch.sum(filt * filt, dim=[1, 2, 3], keepdim=True) + eps)

    def _get_input_norm(self, feat, ksize, pad):
        eps = 1e-4
        channels = feat.shape[1]  # PyTorch uses (batch_size, channels, height, width) format
        shape = (1, channels, ksize, ksize)  # PyTorch uses (out_channels, in_channels/groups, kH, kW) for filters
        filt = torch.ones(shape, device=feat.device)
        input_norm = torch.sqrt(F.conv2d(feat*feat, filt, padding=pad) + eps)
        return input_norm
            
    def forward(self, x):
        # Encoder
        x = F.relu(self.encode_conv1(x))
        x = F.relu(self.encode_conv2(x))
        x = self.encode_conv3(x)

        # Decoder
        x = F.relu(self.decode_convT1(x))
        x = F.relu(self.decode_convT2(x))
        x = torch.tanh(self.decode_convT3(x))

        return x

    def forward_no_norm(self, x):
        # Encoder
        x = F.relu(self.encode_conv1(x))
        x = F.relu(self.encode_conv2(x))
        x = self.encode_conv3(x)

        #print('x', x.shape)

        filt = self.encode_conv3.weight
        #print('filt', filt.shape)
        xnorm = self._get_input_norm(x, ksize=3, pad=1)
        wnorm = self._get_filter_norm(filt)
        wnorm = wnorm.permute(1, 0, 2, 3)

        x = x / xnorm
        x = x / wnorm
        x = x * 60

        # Decoder
        x = F.relu(self.decode_convT1(x))
        x = F.relu(self.decode_convT2(x))
        x = torch.tanh(self.decode_convT3(x))

        return x

    def forward_no_angle(self, x):
        # Encoder
        x = F.relu(self.encode_conv1(x))
        x = F.relu(self.encode_conv2(x))
        x = self.encode_conv3(x)

        filt = self.encode_conv3.weight

        xnorm = self._get_input_norm(x, ksize=3, pad=1)
        wnorm = self._get_filter_norm(filt)
        wnorm = wnorm.permute(1, 0, 2, 3)

        x = xnorm * wnorm
        #x = F.relu(x)

        # Decoder
        x = F.relu(self.decode_convT1(x))
        x = F.relu(self.decode_convT2(x))
        x = torch.tanh(self.decode_convT3(x))

        return x

def unnormalize(img):
    img = img * 0.5 + 0.5  # reverse of normalization operation
    img = img.clamp(0, 1)  # clamp values to be between 0 and 1
    return img

# save decoded images        
def save_decoded_image(img, output_dir, name):
    img = img.view(img.size(0), 3, 64, 64)
    img = unnormalize(img)
    output_path = os.path.join(output_dir, name + '.png')
    save_image(img, output_path)
    
def test_image_reconstruction(net, testloader, output_dir):
    nrow = 4
    outputs_list = []
    for i, img in enumerate(testloader):
        img = img.to(device)
        outputs = net(img)
        outputs = outputs.view(outputs.size(0), 3, 64, 64).cpu().data
        outputs_list.append(outputs)
        if i >= 15:  # Adjust this value to your preference
            break

    outputs_tensor = torch.cat(outputs_list)
    outputs_tensor = unnormalize(outputs_tensor)
    output_path = os.path.join(output_dir, 'conv_pokemon_reconstruction2.png')
    save_image(outputs_tensor, output_path, nrow=nrow)

def test_image_reconstruction_no_norm(net, testloader, output_dir):
    nrow = 4
    outputs_list = []
    for i, img in enumerate(testloader):
        img = img.to(device)
        outputs = net.forward_no_norm(img)
        outputs = outputs.view(outputs.size(0), 3, 64, 64).cpu().data
        outputs = unnormalize(outputs)
        outputs_list.append(outputs)
        if i >= 15:  # Adjust this value to your preference
            break

    outputs_tensor = torch.cat(outputs_list)
    output_path = os.path.join(output_dir, 'conv_pokemon_reconstruction_no_norm2.png')
    save_image(outputs_tensor, output_path, nrow=nrow)


def test_image_reconstruction_no_angle(net, testloader, output_dir):
    nrow = 4
    outputs_list = []
    for i, img in enumerate(testloader):
        img = img.to(device)
        outputs = net.forward_no_angle(img)
        outputs = outputs.view(outputs.size(0), 3, 64, 64).cpu().data
        outputs = unnormalize(outputs)
        outputs_list.append(outputs)
        if i >= 15:  # Adjust this value to your preference
            break

    outputs_tensor = torch.cat(outputs_list)
    output_path = os.path.join(output_dir, 'conv_pokemon_reconstruction_no_angle2.png')
    save_image(outputs_tensor, output_path, nrow=nrow)

def train(net, trainloader, NUM_EPOCHS, output_dir, model_path):
    train_loss = []
    for epoch in tqdm(range(NUM_EPOCHS)):
        start = time.time()
        running_loss = 0.0
        for img in trainloader:
            img = img.to(device)
            optimizer.zero_grad()
            outputs = net(img)
            loss = criterion(outputs, img)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        loss = running_loss / len(trainloader)
        train_loss.append(loss)
        end = time.time()
        total_time = end - start
        print('- Epoch {} of {}, ETA: {:.2f} Train Loss: {:.3f}'.format(
            epoch+1, NUM_EPOCHS, total_time, loss))
        
        if epoch % 20 == 0:
            save_decoded_image(img.cpu().data, output_dir, name='original{}'.format(epoch))
            save_decoded_image(outputs.cpu().data, output_dir, name='decoded{}'.format(epoch))


if __name__ == "__main__":
    NUM_EPOCHS = 200
    LEANRING_RATE = 0.001
    BATCH_SIZE = 128

    output_dir = './log/ablation_ae'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model_path = os.path.join(output_dir, 'model.pth')
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = AutoEncoder().to(device)
    model.load_state_dict(torch.load(model_path))

    # Apply the transformation to the dataset
    # dataset = load_dataset("lambdalabs/pokemon-blip-captions", split="train")

    transform = transforms.Compose([
        transforms.Resize(size=(64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    # Data loader
    dataset = ImageDataset(dir_path='./data/102flowers/jpg', transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr= LEANRING_RATE)

    # first train the simple convolutional autoencoder
    train(model, dataloader, NUM_EPOCHS, output_dir, model_path)
    torch.save(model.state_dict(), model_path)

    # run inference for reconstructing the input image
    test_image_reconstruction(model, test_loader, output_dir)
    test_image_reconstruction_no_norm(model, test_loader, output_dir)
    test_image_reconstruction_no_angle(model, test_loader, output_dir)

