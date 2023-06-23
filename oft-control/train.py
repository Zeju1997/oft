from oldm.hack import disable_verbosity
disable_verbosity()

import os
import sys
import torch

file_path = os.path.abspath(__file__)
parent_dir = os.path.abspath(os.path.dirname(file_path) + '/..')
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from oldm.logger import ImageLogger
from oldm.model import create_model, load_state_dict
from datasets.utils import return_dataset

from oft import inject_trainable_oft, inject_trainable_oft_conv, inject_trainable_oft_extended, inject_trainable_oft_with_norm

import argparse


parser = argparse.ArgumentParser()

parser.add_argument('--r', type=int, default=4)
parser.add_argument('--eps', type=float, default=1e-5)
parser.add_argument('--coft', action="store_true")
parser.add_argument('--block_share', action="store_true", default=False)

args = parser.parse_args()


if __name__ == "__main__":
    # specify the control signal and dataset
    control = 'densepose' # segm, sketch, densepose, depth, canny, canny1k, canny5k, canny20k, canny50k

    # create dataset
    train_dataset, val_dataset, data_name, logger_freq, max_epochs = return_dataset(control) # , n_samples=n_samples)

    # Configs
    resume_path = './models/control_sd15_ini_oft.ckpt'
    
    batch_size = 16
    num_samples = 6
    plot_frequency = 100
    learning_rate = 1e-5
    sd_locked = True
    only_mid_control = False
    num_gpus = torch.cuda.device_count()
    experiment = 'oft_{}_{}_eps_{}_pe_diff_mlp_r_{}_{}gpu'.format(data_name, control, args.eps, args.r, num_gpus)
    print(experiment)
    sys.exit()

    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = create_model('./configs/oft_ldm_v15.yaml').cpu()
    model.model.requires_grad_(False)
    
    # inject trainable oft parameters
    unet_lora_params, train_names = inject_trainable_oft(model.model, r=args.r, eps=args.eps, is_coft=args.coft, block_share=args.block_share)
    # unet_lora_params, train_names = inject_trainable_oft_conv(model.model, r=args.r, eps=args.eps, is_coft=args.coft, block_share=args.block_share)
    # unet_lora_params, train_names = inject_trainable_oft_extended(model.model, r=args.r, eps=args.eps, is_coft=args.coft, block_share=args.block_share)
    
    model.load_state_dict(load_state_dict(resume_path, location='cpu'))
    model.learning_rate = learning_rate
    model.sd_locked = sd_locked
    model.only_mid_control = only_mid_control

    checkpoint_callback = ModelCheckpoint(
        dirpath='log/image_log_' + experiment,
        filename='model-{epoch:02d}',
        save_top_k=-1,
        save_last=True,
        every_n_epochs=1,
        monitor=None,  # No specific metric to monitor for saving
    )

    # Misc
    train_dataloader = DataLoader(train_dataset, num_workers=0, batch_size=batch_size, shuffle=False)
    val_dataloader = DataLoader(val_dataset, num_workers=0, batch_size=1, shuffle=False)

    logger = ImageLogger(
        val_dataloader=val_dataloader,
        batch_frequency=logger_freq, 
        experiment=experiment, 
        plot_frequency=plot_frequency,
        num_samples=num_samples,
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        gpus=num_gpus, 
        precision=32, 
        callbacks=[logger, checkpoint_callback],
    )

    # Train!
    trainer.fit(model, train_dataloader)
