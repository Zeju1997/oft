import os
import logging
import itertools
import numpy as np
import random
import einops
import torch
import torchvision
from PIL import Image
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.distributed import rank_zero_only

import matplotlib.pyplot as plt


class ImageLogger(Callback):
    def __init__(self, val_dataloader=None, batch_frequency=2000, max_images=4, clamp=True, increase_log_steps=True,
                 rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False,
                 log_images_kwargs=None, experiment=None, plot_frequency=300, num_samples=1):
        super().__init__()
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step
        self.experiment = experiment
        self.log_path = os.path.join("log", "image_log_{}".format(self.experiment))

        self.init_loss_logger()
        self.training_losses = []
        self.global_steps = []
        self.plot_frequency = plot_frequency
        self.val_dataloader = val_dataloader
        self.num_samples = num_samples

    def init_loss_logger(self):
        os.makedirs(self.log_path, exist_ok=True)
        logging.basicConfig(filename=os.path.join(self.log_path, "training_loss.log"), level=logging.INFO,
                            format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    @rank_zero_only
    def log_loss(self, loss, global_step):
        logging.info(f"Step: {global_step}, Loss: {loss}")
        self.training_losses.append(loss)
        self.global_steps.append(global_step)

    @rank_zero_only
    def update_loss_plot(self):
        plt.figure()
        plt.plot(self.global_steps, self.training_losses, label="Training loss")
        plt.xlabel("Global Step")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(self.log_path, "training_loss_plot.png"))
        plt.close()

    @rank_zero_only
    def log_local(self, save_dir, split, images, global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, "log", "image_log_{}".format(self.experiment), split)
        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=4)
            if self.rescale:
                grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid * 255).astype(np.uint8)
            filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(k, global_step, current_epoch, batch_idx)
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)

    @rank_zero_only
    def log_local_val(self, save_dir, split, images, global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, "log", "image_log_{}".format(self.experiment), split, str(current_epoch))
        for k in images:
            for idx, image in enumerate(images[k]):
                if self.rescale:
                    image = (image + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
                image = image.transpose(0, 1).transpose(1, 2).squeeze(-1)
                image = image.numpy()
                image = (image * 255).astype(np.uint8)
                filename = "{}_gs-{:06}_e-{:06}_b-{:06}_i-{:06}.png".format(k, global_step, current_epoch, batch_idx, idx)
                path = os.path.join(root, filename)
                os.makedirs(os.path.split(path)[0], exist_ok=True)
                Image.fromarray(image).save(path)

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        check_idx = batch_idx  # if self.log_on_batch_idx else pl_module.global_step
        if (self.check_frequency(check_idx) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0):
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images(batch, split=split, **self.log_images_kwargs)

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1., 1.)

            self.log_local(pl_module.logger.save_dir, split, images,
                           pl_module.global_step, pl_module.current_epoch, batch_idx)

            if is_train:
                pl_module.train()

    def log_img_val(self, pl_module, batch, batch_idx, split="val"):
        check_idx = batch_idx  # if self.log_on_batch_idx else pl_module.global_step
        if (hasattr(pl_module, "log_images") and callable(pl_module.log_images) and self.max_images > 0):
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images(batch, split=split, num_samples=self.num_samples, **self.log_images_kwargs)

            for k in images:
                N = min(images[k].shape[0], self.num_samples)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1., 1.)

            self.log_local_val(pl_module.logger.save_dir, split, images,
                           pl_module.global_step, pl_module.current_epoch, batch_idx)

            if is_train:
                pl_module.train()

    def check_frequency(self, check_idx):
        return check_idx % self.batch_freq == 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if not self.disabled:
            self.log_img(pl_module, batch, batch_idx, split="train")

        '''
        if "loss" in outputs:
            train_loss = outputs["loss"]
            if isinstance(train_loss, torch.Tensor):
                train_loss = train_loss.detach().cpu().numpy()
            self.log_loss(train_loss, pl_module.global_step)

            if pl_module.global_step % self.plot_frequency == 0:
                self.update_loss_plot()
        '''
        
    def on_epoch_end(self, trainer, pl_module):
        if not self.disabled:
            for batch_idx, batch in enumerate(self.val_dataloader):
                input_image, prompt, hint = batch['jpg'], batch['txt'], batch['hint']
                self.log_img_val(pl_module, batch, batch_idx, split="val")

'''
class ImageLogger(Callback):
    def __init__(self, batch_frequency=2000, max_images=4, clamp=True, increase_log_steps=True,
                 rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False,
                 log_images_kwargs=None, experiment=None):
        super().__init__()
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step
        self.experiment = experiment

        self.init_tensorboard_logger()

    def init_tensorboard_logger(self):
        self.logger = TensorBoardLogger("log/image_log_{}".format(self.experiment), name="my_experiment")

    @rank_zero_only
    def log_local(self, save_dir, split, images, global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, "log", "image_log_{}".format(self.experiment), split)
        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=4)
            if self.rescale:
                grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid * 255).astype(np.uint8)
            filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(k, global_step, current_epoch, batch_idx)
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        check_idx = batch_idx  # if self.log_on_batch_idx else pl_module.global_step
        if (self.check_frequency(check_idx) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0):
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images(batch, split=split, **self.log_images_kwargs)

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1., 1.)

            self.log_local(pl_module.logger.save_dir, split, images,
                           pl_module.global_step, pl_module.current_epoch, batch_idx)

            if is_train:
                pl_module.train()

    def log_hyperparams(self, *args, **kwargs):
        self.logger.log_hyperparams(*args, **kwargs)

    def log_graph(self, *args, **kwargs):
        self.logger.log_graph(*args, **kwargs)

    def save(self):
        self.logger.save()

    @property
    def save_dir(self):
        return self.logger.save_dir

    def check_frequency(self, check_idx):
        return check_idx % self.batch_freq == 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if not self.disabled:
            self.log_img(pl_module, batch, batch_idx, split="train")

        if "loss" in outputs:
            train_loss = outputs["loss"]
            if isinstance(train_loss, torch.Tensor):
                train_loss = train_loss.detach().cpu().numpy()
            self.logger.experiment.add_scalar("Loss/train", train_loss, pl_module.global_step)
'''