from abc import abstractmethod
import math

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch as th
from torch import einsum
import torch

from einops import rearrange, repeat

from ldm.modules.diffusionmodules.util import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)
from ldm.modules.diffusionmodules.openaimodel import TimestepBlock
from ldm.util import exists
from ldm.modules.attention import (
    CrossAttention, 
    Normalize, 
    MemoryEfficientCrossAttention, 
    default, 
    BasicTransformerBlock,
    FeedForward,
    GEGLU,
)

try:
    import xformers
    import xformers.ops
    XFORMERS_IS_AVAILBLE = True
except:
    XFORMERS_IS_AVAILBLE = False

# CrossAttn precision handling
import os
_ATTN_PRECISION = os.environ.get("ATTN_PRECISION", "fp32")


class ResBlock_opt(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            # conv_nd(dims, channels, self.out_channels, 3, padding=1),
            # OPT_cp(channels, self.out_channels, 3, padding=1),
            # OPT_sgs(channels, self.out_channels, 3, padding=1),
            OPT_pe(channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                # conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1),
                # OPT_cp(self.out_channels, self.out_channels, 3, padding=1),
                # OPT_sgs(self.out_channels, self.out_channels, 3, padding=1),
                OPT_pe(self.out_channels, self.out_channels, 3, padding=1),
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )


    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


def project(R, eps):
    I = torch.zeros((R.size(0), R.size(0)), dtype=R.dtype, device=R.device)
    diff = R - I
    norm_diff = torch.norm(diff)
    if norm_diff <= eps:
        return R
    else:
        #print('projecting !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        return I + eps * (diff / norm_diff)

def project_batch(R, eps=1e-5):
    # scaling factor for each of the smaller block matrix
    eps = eps * 1 / torch.sqrt(torch.tensor(R.shape[0]))
    I = torch.zeros((R.size(1), R.size(1)), device=R.device, dtype=R.dtype).unsqueeze(0).expand_as(R)
    diff = R - I
    norm_diff = torch.norm(R - I, dim=(1, 2), keepdim=True)
    mask = (norm_diff <= eps).bool()
    #if not mask.any():
    #    print("projecting batch!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    out = torch.where(mask, R, I + eps * (diff / norm_diff))
    return out


class OPT_cp(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, epsilon=None):
        super(OPT_cp, self).__init__()

        self.in_channels=in_channels
        self.out_channels=out_channels
        self.kernel_size=kernel_size
        self.stride=stride
        self.padding=padding

        # Define the fixed Conv2d layer: v
        self.OPT = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.filt_shape = [out_channels, in_channels, kernel_size, kernel_size]
        self.fix_filt_shape = [kernel_size * kernel_size * in_channels, out_channels]

        # Define the trainable matrix parameter: R
        # Initialized as an identity matrix
        self.R_shape = [kernel_size * kernel_size * in_channels, kernel_size * kernel_size * in_channels]
        self.R = nn.Parameter(th.eye(self.R_shape[0]), requires_grad=True)

        self.eps = 1e-8 * self.R_shape[0] * self.R_shape[0]

    def forward(self, x):
        with th.no_grad():
            self.R.copy_(project(self.R, eps=self.eps))

        # Cayley parametrization
        # orth_rotate = self.cayley(self.R)
        orth_rotate = self.R

        # fix filter
        fix_filt = self.OPT.weight.data
        fix_filt = fix_filt.view(self.fix_filt_shape)
        filt = th.mm(orth_rotate, fix_filt)
        filt = filt.view(self.filt_shape)
  
        # Apply the trainable identity matrix
        out = F.conv2d(x, filt, stride=self.stride, padding=self.padding)
        
        return out 

    def cayley(self, data):
        R = data
        r, c = list(R.shape)
        upper = th.triu(R)
        skew = upper + th.transpose(-1*upper, 0, 1)
        I = th.eye(r, device=R.device)
        Q = th.mm(I + skew, th.inverse(I-skew))
        return Q


class OPT_sgs(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, epsilon=None):
        super(OPT_sgs, self).__init__()
        # Get the number of available GPUs
        self.num_gpus = th.cuda.device_count()
        # Set the device IDs for distributed training
        self.device_ids = list(range(self.num_gpus))

        self.in_channels=in_channels
        self.out_channels=out_channels
        self.kernel_size=kernel_size
        self.stride=stride
        self.padding=padding

        self.s=int(0.25*self.in_channels)

        # Define the fixed Conv2d layer: v
        self.OPT = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)

        self.filt_shape = [out_channels, in_channels, kernel_size, kernel_size]
        self.fix_filt_shape = [kernel_size * kernel_size * in_channels, out_channels]

        # Define the trainable matrix parameter: R
        # Initialized as an identity matrix
        self.R_shape = [kernel_size * kernel_size * self.s, kernel_size * kernel_size * self.s]
        self.R = nn.Parameter(th.eye(self.R_shape[0])) #, requires_grad=True)

        self.eps = 1e-8 * self.R_shape[0] * self.R_shape[0]

        self.indices = th.randperm(self.kernel_size*self.kernel_size*self.in_channels)
        self.num_chunks = 4
        self.chunks = th.chunk(self.indices, self.num_chunks)

    def forward(self, x):
        with th.no_grad():
            self.R.copy_(project(self.R, eps=self.eps))

        # Gram Schmidt parametrization
        orth_rotate = self.cayley(self.R)
        # orth_rotate = self.R
        # print('is orthogonal', self.is_orthogonal(orth_rotate))

        # fix filter
        fix_filt = self.OPT.weight.data
        fix_filt = fix_filt.view(self.fix_filt_shape)

        # sparse fix filter
        i = th.randperm(self.num_chunks)[0]
        idx = self.chunks[i]
        # idx = th.from_numpy(idx).long()
        fix_s = fix_filt[idx]
        updates = th.mm(orth_rotate, fix_s)
        fix_filt[idx] = updates
        filt = fix_filt.view(self.filt_shape)
  
        # Apply the trainable identity matrix
        out = F.conv2d(x, filt, stride=self.stride, padding=self.padding)

        return out 

    def gram_schmidt(self, vectors):
        basis = []
        for v in vectors:
            w = v.clone().detach()
            for b in basis:
                w -= th.dot(w, b) * b
            if th.norm(w) > 1e-10:
                w /= th.norm(w)
                basis.append(w)

        return th.stack(basis, dim=0)

    def cayley(self, data):
        R = data
        r, c = list(R.shape)
        upper = th.triu(R)
        skew = upper + th.transpose(-1*upper, 0, 1)
        I = th.eye(r, device=R.device)
        Q = th.mm(I + skew, th.inverse(I-skew))
        return Q

    def is_orthogonal(self, R, eps=1e-5):
        with th.no_grad():
            RtR = th.matmul(R.t(), R)
            diff = th.abs(RtR - th.eye(R.shape[1], dtype=R.dtype, device=R.device))
            return th.all(diff < eps)



class OPT_pe(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, epsilon=None, dim=3):
        super(OPT_pe, self).__init__()
        # Get the number of available GPUs
        self.num_gpus = th.cuda.device_count()
        # Set the device IDs for distributed training
        self.device_ids = list(range(self.num_gpus))

        self.in_channels=in_channels
        self.out_channels=out_channels
        self.kernel_size=kernel_size
        self.stride=stride
        self.padding=padding

        # Define the fixed Conv2d layer: v
        self.OPT = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        #for param in self.OPT.parameters():
        #    param.requires_grad = False

        self.filt_shape = [out_channels, in_channels, kernel_size, kernel_size]
        self.fix_filt_shape = [kernel_size * kernel_size * in_channels, out_channels]

        eps = 1e-3
        # Define the trainable matrix parameter: R
        self.dim = dim
        if self.dim == 2:
            # Initialized as an identity matrix
            self.R_shape = [kernel_size * kernel_size, kernel_size * kernel_size]
            self.R = nn.Parameter(th.zeros(self.R_shape[0], self.R_shape[0]), requires_grad=True)
  
            self.eps = eps * self.R_shape[0] * self.R_shape[0]
        else:
            # Initialized as an identity matrix
            self.R_shape = [in_channels, kernel_size * kernel_size, kernel_size * kernel_size]
            R = th.zeros(self.R_shape[1], self.R_shape[1])
            R = th.stack([R] * self.in_channels)
            self.R = nn.Parameter(R, requires_grad=True)

            self.eps = eps * self.R_shape[1] * self.R_shape[1]

    def forward(self, x):
        if self.dim == 2:
            with th.no_grad():
                self.R.copy_(project(self.R, eps=self.eps))
            orth_rotate = self.cayley(self.R)
        else:
            with th.no_grad():
                self.R.copy_(project_batch(self.R, eps=self.eps))
            orth_rotate = self.cayley_batch(self.R)

        # Gram Schmidt parametrization
        # orth_rotate = self.cayley(self.R)
        # orth_rotate = self.R
        # print('is orthogonal', self.is_orthogonal(orth_rotate))

        # Block-diagonal parametrization
        block_diagonal_matrix = self.block_diagonal(orth_rotate)

        # fix filter
        fix_filt = self.OPT.weight.data
        fix_filt = fix_filt.view(self.fix_filt_shape)
        filt = th.mm(block_diagonal_matrix, fix_filt)
        filt = filt.view(self.filt_shape)

        # Apply the trainable identity matrix
        bias_term = self.OPT.bias.data if self.OPT.bias is not None else None
        out = F.conv2d(input=x, weight=filt, bias=bias_term, stride=self.stride, padding=self.padding)
        # out = self.OPT(x)
        
        return out 

    def cayley(self, data):
        R = data
        r, c = list(R.shape)
        # Ensure the input matrix is skew-symmetric
        skew = 0.5 * (R - R.t())
        I = torch.eye(r, device=R.device)
        # Perform the Cayley parametrization
        Q = torch.mm(I + skew, torch.inverse(I - skew))
        
        return Q

    def cayley_batch(self, data):
        B, r, c = data.shape
        # Ensure the input matrix is skew-symmetric
        skew = 0.5 * (data - data.transpose(1, 2))
        I = th.eye(r, device=data.device).unsqueeze(0).repeat(B, 1, 1)
        # Perform the Cayley parametrization
        Q = th.bmm(I + skew, th.inverse(I - skew))

        return Q


    def block_diagonal(self, R):
        if self.dim == 2:
            # Create a list of R repeated block_count times
            blocks = [R] * self.in_channels
        else:
            # Create a list of R slices along the third dimension
            blocks = [R[i, ...] for i in range(self.in_channels)]

        # Use torch.block_diag to create the block diagonal matrix
        A = th.block_diag(*blocks)

        return A

    def is_orthogonal(self, R, eps=1e-5):
        with th.no_grad():
            RtR = th.matmul(R.t(), R)
            diff = th.abs(RtR - th.eye(R.shape[1], dtype=R.dtype, device=R.device))
            return th.all(diff < eps)

'''

class OPT_pe(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, epsilon=None, dim=3):
        super(OPT_pe, self).__init__()
        # Get the number of available GPUs
        self.num_gpus = th.cuda.device_count()
        # Set the device IDs for distributed training
        self.device_ids = list(range(self.num_gpus))

        self.in_channels=in_channels
        self.out_channels=out_channels
        self.kernel_size=kernel_size
        self.stride=stride
        self.padding=padding

        # Initialize the fixed Conv2d layer's weight and bias
        weight = torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        self.register_buffer('OPT_weight', weight)
        
        if bias:
            bias = torch.empty(out_channels)
            self.register_buffer('OPT_bias', bias)
        else:
            self.register_buffer('OPT_bias', None)

        self.filt_shape = [out_channels, in_channels, kernel_size, kernel_size]
        self.fix_filt_shape = [kernel_size * kernel_size * in_channels, out_channels]

        eps = 1e-2
        # Define the trainable matrix parameter: R
        self.dim = dim
        if self.dim == 2:
            # Initialized as an identity matrix
            self.R_shape = [kernel_size * kernel_size, kernel_size * kernel_size]
            self.R = nn.Parameter(th.eye(self.R_shape[0]), requires_grad=True)
  
            self.eps = eps * self.R_shape[0] * self.R_shape[0]
        else:
            # Initialized as an identity matrix
            self.R_shape = [in_channels, kernel_size * kernel_size, kernel_size * kernel_size]
            R = th.eye(self.R_shape[1])
            R = th.stack([R] * self.in_channels)
            self.R = nn.Parameter(R, requires_grad=True)

            self.eps = eps * self.R_shape[1] * self.R_shape[1]

    def forward(self, x):
        if self.dim == 2:
            with th.no_grad():
                self.R.copy_(project(self.R, eps=self.eps))
            orth_rotate = self.cayley(self.R)
        else:
            with th.no_grad():
                self.R.copy_(project_batch(self.R, eps=self.eps))
            orth_rotate = self.cayley_batch(self.R)

        # Gram Schmidt parametrization
        # orth_rotate = self.cayley(self.R)
        # orth_rotate = self.R
        # print('is orthogonal', self.is_orthogonal(orth_rotate))

        # Block-diagonal parametrization
        block_diagonal_matrix = self.block_diagonal(orth_rotate)

        # fix filter
        fix_filt = self.OPT_weight
        fix_filt = fix_filt.view(self.fix_filt_shape)
        filt = th.mm(block_diagonal_matrix, fix_filt)
        filt = filt.view(self.filt_shape)

        # Apply the trainable identity matrix
        bias_term = self.OPT_bias if self.OPT_bias is not None else None # self.OPT.bias.data if self.OPT.bias is not None else None
        out = F.conv2d(input=x, weight=filt, bias=bias_term, stride=self.stride, padding=self.padding)
        # out = self.OPT(x)
        
        return out 

    def cayley(self, data):
        R = data
        r, _ = list(R.shape)
        upper = th.triu(R)
        skew = upper + th.transpose(-1*upper, 0, 1)
        I = th.eye(r, device=R.device)
        Q = th.mm(I + skew, th.inverse(I-skew))
        return Q

    def cayley_batch(self, x):
        # x has shape (batch_size, 3, 3)
        b, r, _ = x.shape
        upper = th.triu(x)
        I = th.eye(r, dtype=x.dtype, device=x.device)
        skew = upper + th.transpose(-1*upper, 1, 2)
        Q = th.bmm(I + skew, th.inverse(I - skew))
        #Q_flat = torch.flatten(Q, start_dim=1)
        #print(Q_flat.shape, x.shape)
        #out = torch.bmm(Q_flat.unsqueeze(1), x).view(batch_size, -1)
        return Q


    def block_diagonal(self, R):
        if self.dim == 2:
            # Create a list of R repeated block_count times
            blocks = [R] * self.in_channels
        else:
            # Create a list of R slices along the third dimension
            blocks = [R[i, ...] for i in range(self.in_channels)]
            
        # Use torch.block_diag to create the block diagonal matrix
        A = th.block_diag(*blocks)

        return A

    def is_orthogonal(self, R, eps=1e-5):
        with th.no_grad():
            RtR = th.matmul(R.t(), R)
            diff = th.abs(RtR - th.eye(R.shape[1], dtype=R.dtype, device=R.device))
            return th.all(diff < eps)
'''