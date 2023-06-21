from share import *

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
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from datasets.utils import return_dataset

from cldm.opt_lora import inject_trainable_opt, inject_trainable_opt_conv, inject_trainable_opt_extended, inject_trainable_opt_with_norm

import multiprocessing


if __name__ == "__main__":
    # multiprocessing.freeze_support()

    control = 'segm' # segm, sketch, densepose, depth, canny, canny1k, canny5k, canny20k, canny50k

    #import argparse
    #parser = argparse.ArgumentParser()
    #parser.add_argument('--n_samples', type=int, help='Number of samples')
    #args = parser.parse_args()
    #n_samples = args.n_samples

    # create dataset
    train_dataset, val_dataset, data_name, logger_freq, max_epochs = return_dataset(control) # , n_samples=n_samples)
    max_epochs = 5

    # Configs
    # resume_path = './models/control_sd15_ini_opt_lora_conv.ckpt' # control_sd15_ini, control_sd15_ini_r4

    batch_size = 16 # 4
    num_samples = 6
    plot_frequency = 100
    learning_rate = 1e-5 # 1e-5, 5e-6, 1e-6
    sd_locked = True
    only_mid_control = False
    num_gpus = torch.cuda.device_count()

    # experiment = 'opt_lora_{}_{}_lr_1-5_pe_diff_mlp_r_4_cayley_{}gpu'.format(data_name, control, num_gpus)
    # experiment = 'opt_lora_{}_{}_eps_1-3_pe_diff_mlp_r_4_cayley_{}gpu'.format(data_name, control, num_gpus)    
    # experiment = 'ablation3_{}_opt_lora_{}_{}_eps_1-4_pe_diff_mlp_r_4_cayley_{}gpu'.format(n_samples, data_name, control, num_gpus)
    experiment = 'log/image_log_opt_lora_{}_{}_eps_1-3_pe_diff_mlp_r_4_cayley_{}gpu'.format(data_name, control, num_gpus)    
    epoch = 19

    resume_path = os.path.join(experiment, f'model-epoch={epoch:02d}.ckpt')

    # epoch = 15
    # resume_from_checkpoint_path = os.path.join('log/image_log_'+experiment, f'model-epoch={epoch:02d}.ckpt')

    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = create_model('./models/opt_lora_ldm_v15.yaml').cpu()
    model.model.requires_grad_(False)
    # unet_opt_params, train_names = inject_trainable_opt(model.model)
    # unet_opt_params, train_names = inject_trainable_opt_extended(model.model)
    # unet_opt_params, train_names = inject_trainable_opt_conv(model.model)
    unet_opt_params, train_names = inject_trainable_opt_with_norm(model.model)

    for name, param in model.named_parameters():
        if 'scaling_factors' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False


    #with open('model_names_with_norm.txt', 'w') as f:
    #    for name, param in model.named_parameters():
    #        if param.requires_grad:
    #            f.write(str(name) + '\n')

    model.load_state_dict(load_state_dict(resume_path, location='cpu'), strict=False)
    model.learning_rate = learning_rate
    model.sd_locked = sd_locked
    model.only_mid_control = only_mid_control

    #experiment1 = 'opt_lora_{}_{}_eps_1-3_pe_diff_mlp_r_4_cayley_with_norm_{}gpu'.format(data_name, control, num_gpus)    

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
        # resume_from_checkpoint=resume_from_checkpoint_path,  # Add this line
        # logger=logger,
        # stragtegy="ddp",
    )

    # Train!
    trainer.fit(model, train_dataloader)
