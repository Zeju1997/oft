import os
from glob import glob
from tqdm import tqdm
import numpy as np
from pathlib import Path
from skimage.io import imread, imsave
import cv2
import json

end_list = np.array([17, 22, 27, 42, 48, 31, 36, 68], dtype = np.int32) - 1
def plot_kpts(image, kpts, color = 'g'):
    ''' Draw 68 key points
    Args: 
        image: the input image
        kpt: (68, 3).
    '''
    if color == 'r':
        c = (255, 0, 0)
    elif color == 'g':
        c = (0, 255, 0)
    elif color == 'b':
        c = (255, 0, 0)
    image = image.copy()
    kpts = kpts.copy()
    radius = max(int(min(image.shape[0], image.shape[1])/200), 1)
    for i in range(kpts.shape[0]):
        st = kpts[i, :2]
        if kpts.shape[1]==4:
            if kpts[i, 3] > 0.5:
                c = (0, 255, 0)
            else:
                c = (0, 0, 255)
        image = cv2.circle(image,(int(st[0]), int(st[1])), radius, c, radius*2)  
        if i in end_list:
            continue
        ed = kpts[i + 1, :2]
        image = cv2.line(image, (int(st[0]), int(st[1])), (int(ed[0]), int(ed[1])), (255, 255, 255), radius)
    return image

def plot_points(image, kpts, color = 'w'):
    ''' Draw 68 key points
    Args: 
        image: the input image
        kpt: (n, 3).
    '''
    if color == 'r':
        c = (255, 0, 0)
    elif color == 'g':
        c = (0, 255, 0)
    elif color == 'b':
        c = (0, 0, 255)
    elif color == 'y':
        c = (0, 255, 255)
    elif color == 'w':
        c = (255, 255, 255)
    image = image.copy()
    kpts = kpts.copy()
    kpts = kpts.astype(np.int32)
    radius = max(int(min(image.shape[0], image.shape[1])/200), 1)
    for i in range(kpts.shape[0]):
        st = kpts[i, :2]
        image = cv2.circle(image,(int(st[0]), int(st[1])), radius, c, radius*2)  
    return image

def generate_landmark2d(inputpath, savepath, n=0, device='cuda:0', vis=False):
    print(f'generate 2d landmarks')
    os.makedirs(savepath, exist_ok=True)
    import face_alignment
    detect_model = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device=device, flip_input=False)
    
    imagepath_list = glob(os.path.join(inputpath, '*_{}.png'.format(n)))
    imagepath_list = sorted(imagepath_list)
    for imagepath in tqdm(imagepath_list):
        name = Path(imagepath).stem

        image = imread(imagepath)[:,:,:3]
        out = detect_model.get_landmarks(image)
        if out is None:
            continue
        kpt = out[0].squeeze()
        np.savetxt(os.path.join(savepath, f'{name}.txt'), kpt)
        if vis:
            image = cv2.imread(imagepath)
            image_point = plot_kpts(image, kpt)
            # check
            cv2.imwrite(os.path.join(savepath, f'{name}_overlay.jpg'), image_point)
            # background = np.zeros_like(image)
            # cv2.imwrite(os.path.join(savepath, f'{name}_line.jpg'), plot_kpts(background, kpt))
            # cv2.imwrite(os.path.join(savepath, f'{name}_point.jpg'), plot_points(background, kpt))
        # exit()

def landmark_comparison(lmk_folder, gt_lmk_folder, n=0):
    print(f'calculate reprojection error')
    lmk_err = []
    gt_lmk_folder = './data/celebhq-text/celeba-hq-landmark2d'
    with open('./data/celebhq-text/prompt_val_blip_full.json', 'rt') as f: # fill50k, COCO
        for line in f:
            val_data = json.loads(line)
    for i in tqdm(range(2000)):
        # import ipdb; ipdb.set_trace()
        line = val_data[n]

        img_name = line["image"][:-4]
        lmk1_path = os.path.join(gt_lmk_folder, f'{img_name}.txt')

        lmk1 = np.loadtxt(lmk1_path) / 2
        lmk2_path = os.path.join(lmk_folder, f'result_{i}_{n}.txt')
        if not os.path.exists(lmk2_path):
            print(f'{lmk2_path} not exist')
            continue
        lmk2 = np.loadtxt(lmk2_path)
        lmk_err.append(np.mean(np.linalg.norm(lmk1 - lmk2, axis=1)))
    print(np.mean(lmk_err))
    np.save(os.path.join(lmk_folder, 'lmk_err.npy'), lmk_err)


n = 0
epoch = 19
gt_lmk_folder = './data/celebhq-text/celeba-hq-landmark2d'
input_folder = os.path.join('./data/image_log_opt_lora_CelebA_landmark_lr_5-6_pe_diff_mlp_r_4_cayley_4gpu/results', str(epoch))
save_folder = os.path.join(input_folder, 'landmark')

generate_landmark2d(input_folder, save_folder, n, device='cuda:0', vis=False)
landmark_comparison(save_folder, gt_lmk_folder, n)