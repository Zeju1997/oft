from PIL import Image
from tqdm import tqdm
import sys
import os
import json
import cv2
import torch
from torch.autograd import Variable
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CocoCaptions
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
from torchvision.utils import save_image
import numpy as np

# from net_canny import Net
from ControlNet.annotator.canny import CannyDetector
from ControlNet.annotator.util import resize_image, HWC3

# from pytorch_msssim import ssim, ms_ssim, SSIM

class ResultFolderDataset(Dataset):
    def __init__(self, data_dir, results_dir, n, transform=None):
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.n = n
        self.image_paths = sorted([f for f in os.listdir(data_dir) if f.lower().endswith(('.png'))])
        # self.image_paths = sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith('_{}.png'.format(n))]) 
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_name = self.image_paths[idx]
        source_path = os.path.join(self.data_dir, image_name)

        base_name = image_name.split('_')[1].split('.')[0]  # Extract 'x' from 'image_x.png'
        image_name2 = f'result_{base_name}_{self.n}.png'
        result_path = os.path.join(self.results_dir, image_name2)

        source_image = Image.open(source_path) #.convert('RGB')
        result_image = Image.open(result_path) #.convert('RGB')
        if self.transform:
            source_image = self.transform(source_image)
            result_image = self.transform(result_image)
        return source_image, result_image


def calculate_metrics(pred, target):
    intersection = np.logical_and(pred, target)
    union = np.logical_or(pred, target)
    iou_score = np.sum(intersection) / np.sum(union)
    
    accuracy = np.sum(pred == target) / target.size
    
    tp = np.sum(intersection)  # True positive
    fp = np.sum(pred) - tp  # False positive
    fn = np.sum(target) - tp  # False negative
    
    f1_score = (2 * tp) / (2 * tp + fp + fn)
    
    return iou_score, accuracy, f1_score

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    low_threshold = 100
    high_threshold = 200

    n = 0
    epoch = 9
    # experiment = './log/image_log_lora_COCO_canny_8gpu' # [IOU]: 0.16249358979539114 [Accuracy]: 0.8426783195495605 [F1 Score]: 0.278544679272223
    # experiment = './log/image_log_cldm_COCO_canny_4gpu' # [IOU]: 0.18977163663330127 [Accuracy]: 0.848788190460205 [F1 Score]: 0.3178471245560538
    # experiment = './log/t2i_canny_4gpu' # [IOU]: 0.07803869407915275 [Accuracy]: 0.8239055061340332 [F1 Score]: 0.14330854234781343
    # experiment = './log/sd_canny' # [IOU]: 0.049724708443740134 [Accuracy]: 0.8244535125732422 [F1 Score]: 0.09328981588306719
    experiment = './log/image_log_opt_lora_COCO_canny_eps_1-3_pe_diff_mlp_r_4_cayley_4gpu' # [FID] 27.707968541924515 [IOU]: 0.1953904951867336 [Accuracy]: 0.8430354614257812 [F1 Score]: 0.3255332740111943
    # experiment = './log/image_log_opt_lora_COCO_canny_lr_1e-05_pe_diff_mlp_r_4_cayley_4gpu' # [FID] 27.707968541924515 [IOU]: 0.1953904951867336 [Accuracy]: 0.8430354614257812 [F1 Score]: 0.3255332740111943
    # experiment = './log/image_log_opt_lora_COCO_canny_eps_1-2_pe_diff_mlp_r_4_cayley_4gpu' # [FID] 27.707968541924515 [IOU]: 0.1953904951867336 [Accuracy]: 0.8430354614257812 [F1 Score]: 0.3255332740111943
    
    data_dir = os.path.join(experiment, 'source', str(epoch))
    result_dir = os.path.join(experiment, 'results', str(epoch))
    json_file = os.path.join(experiment, 'results.json')

    # Define the transforms to apply to the images
    transform = transforms.Compose([
        # transforms.Resize((512, 512)),
        # transforms.CenterCrop(512),
        transforms.ToTensor()
    ])

    dataset = ResultFolderDataset(data_dir, result_dir, n=n, transform=transform)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # prompts = []

    # use_cuda = True
    # net = Net(thres10hold=3.0, use_cuda=use_cuda)
    # net = net.to(device)
    # net.eval()

    apply_canny = CannyDetector()

    loss = 0
    iou_score_mean = 0
    accuracy_mean = 0
    f1_score_mean = 0
    ssim_mean = 0
    for i, data in tqdm(enumerate(data_loader)):
        source_image, result_image = data
        # Convert the tensor to a numpy array and transpose it to have the channels last (H, W, C)
        source_image_np = source_image.squeeze(0).permute(1, 2, 0).numpy()
        result_image_np = result_image.squeeze(0).permute(1, 2, 0).numpy()

        # Convert the image to 8-bit unsigned integers (0-255)
        source_image_np = (source_image_np * 255).astype(np.uint8)
        result_image_np = (result_image_np * 255).astype(np.uint8)

        source_detected_map = apply_canny(source_image_np, low_threshold, high_threshold) / 255
        result_detected_map = apply_canny(result_image_np, low_threshold, high_threshold) / 255

        # l1 loss
        # loss = loss + np.mean(np.abs(source_detected_map - result_detected_map))

        iou_score, accuracy, f1_score = calculate_metrics(result_detected_map, source_detected_map)

        iou_score_mean = iou_score_mean + iou_score
        accuracy_mean = accuracy_mean +  accuracy
        f1_score_mean = f1_score_mean + f1_score

        # source_image_th = torch.from_numpy(source_image_np)
        # result_image_th = torch.from_numpy(result_image_np)
        # ssim_val = ssim(source_image_th.unsqueeze(0), result_image_th.unsqueeze(0), data_range=255, size_average=False)
        # ssim_mean = ssim_mean + ssim_val.item()

        # pixel accuracy
        # accuracy = np.sum(source_detected_map == result_detected_map) / source_detected_map.size
        # loss = loss + accuracy

    iou_score_mean = iou_score_mean / len(dataset)
    accuracy_mean = accuracy_mean / len(dataset)
    f1_score_mean = f1_score_mean / len(dataset)
    # ssim_mean = ssim_mean / len(dataset)

    print(experiment)
    print('[Canny]', '[IOU]:', iou_score_mean, '[F1 Score]:', f1_score_mean) #, '[SSIM]:', ssim_mean)