"""
This script utilizes code from lora available at: 
https://github.com/cloneofsimo/lora/blob/master/lora_diffusion/lora.py

Original Author: Simo Ryu
License: Apache License 2.0
"""



import json
import math
from itertools import groupby
from typing import Callable, Dict, List, Optional, Set, Tuple, Type, Union

import pickle

import numpy as np
import PIL
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from safetensors.torch import safe_open
    from safetensors.torch import save_file as safe_save

    safetensors_available = True
except ImportError:
    from .safe_open import safe_open

    def safe_save(
        tensors: Dict[str, torch.Tensor],
        filename: str,
        metadata: Optional[Dict[str, str]] = None,
    ) -> None:
        raise EnvironmentError(
            "Saving safetensors requires the safetensors library. Please install with pip or similar."
        )

    safetensors_available = False


def project(R, eps):
    I = torch.zeros((R.size(0), R.size(0)), dtype=R.dtype, device=R.device)
    diff = R - I
    norm_diff = torch.norm(diff)
    if norm_diff <= eps:
        return R
    else:
        return I + eps * (diff / norm_diff)

def project_batch(R, eps=1e-5):
    # scaling factor for each of the smaller block matrix
    eps = eps * 1 / torch.sqrt(torch.tensor(R.shape[0]))
    I = torch.zeros((R.size(1), R.size(1)), device=R.device, dtype=R.dtype).unsqueeze(0).expand_as(R)
    diff = R - I
    norm_diff = torch.norm(R - I, dim=(1, 2), keepdim=True)
    mask = (norm_diff <= eps).bool()
    out = torch.where(mask, R, I + eps * (diff / norm_diff))
    return out


class OFTInjectedLinear(nn.Module):
    def __init__(
        self, in_features, out_features, bias=False, r=4, eps=1e-5, is_coft=True, block_share=False,
    ):
        super().__init__()

        assert in_features % r == 0, "in_features must be divisible by r"

        # Get the number of available GPUs
        self.num_gpus = torch.cuda.device_count()
        # Set the device IDs for distributed training
        self.device_ids = list(range(self.num_gpus))

        self.in_features=in_features
        self.out_features=out_features

        # Define the fixed Linear layer: v
        self.OFT = torch.nn.Linear(in_features=in_features, out_features=out_features, bias=bias)

        # Define the reduction rate:
        self.r = r
        self.is_coft = is_coft

        self.fix_filt_shape = [in_features, out_features]

        # Define the trainable matrix parameter: R
        self.block_share = block_share
        if self.block_share:
            # Initialized as an identity matrix
            self.R_shape = [in_features // self.r, in_features // self.r]
            self.R = nn.Parameter(torch.zeros(self.R_shape[0], self.R_shape[0]), requires_grad=True)
  
            self.eps = eps * self.R_shape[0] * self.R_shape[0]
        else:
            # Initialized as an identity matrix
            self.R_shape = [self.r, in_features // self.r, in_features // self.r]
            R = torch.zeros(self.R_shape[1], self.R_shape[1])
            R = torch.stack([R] * self.r)
            self.R = nn.Parameter(R, requires_grad=True)
            self.eps = eps * self.R_shape[1] * self.R_shape[1]

    def forward(self, x):
        orig_dtype = x.dtype
        dtype = self.R.dtype

        if self.block_share:
            if self.is_coft:
                with torch.no_grad():
                    self.R.copy_(project(self.R, eps=self.eps))
            orth_rotate = self.cayley(self.R)
        else:
            if self.is_coft:
                with torch.no_grad():
                    self.R.copy_(project_batch(self.R, eps=self.eps))
            orth_rotate = self.cayley_batch(self.R)

        # Block-diagonal parametrization
        block_diagonal_matrix = self.block_diagonal(orth_rotate)
        
        # fix filter
        fix_filt = self.OFT.weight.data
        fix_filt = torch.transpose(fix_filt, 0, 1)
        filt = torch.mm(block_diagonal_matrix, fix_filt.to(dtype))
        filt = torch.transpose(filt, 0, 1)
 
        # Apply the trainable identity matrix
        bias_term = self.OFT.bias.data if self.OFT.bias is not None else None
        out = nn.functional.linear(input=x, weight=filt, bias=bias_term)

        return out #.to(orig_dtype)

    def cayley(self, data):
        r, c = list(data.shape)
        # Ensure the input matrix is skew-symmetric
        skew = 0.5 * (data - data.t())
        I = torch.eye(r, device=data.device)
        
        # Perform the Cayley parametrization
        Q = torch.mm(I + skew, torch.inverse(I - skew))
        return Q
    
    def cayley_batch(self, data):
        b, r, c = data.shape
        # Ensure the input matrix is skew-symmetric
        skew = 0.5 * (data - data.transpose(1, 2))
        I = torch.eye(r, device=data.device).unsqueeze(0).expand(b, r, c)

        # Perform the Cayley parametrization
        Q = torch.bmm(I + skew, torch.inverse(I - skew))

        return Q

    def block_diagonal(self, R):
        if self.block_share:
            # Create a list of R repeated block_count times
            blocks = [R] * self.r
        else:
            # Create a list of R slices along the third dimension
            blocks = [R[i, ...] for i in range(self.r)]

        # Use torch.block_diag to create the block diagonal matrix
        A = torch.block_diag(*blocks)

        return A

    def is_orthogonal(self, R, eps=1e-5):
        with torch.no_grad():
            RtR = torch.matmul(R.t(), R)
            diff = torch.abs(RtR - torch.eye(R.shape[1], dtype=R.dtype, device=R.device))
            return torch.all(diff < eps)

    def is_identity_matrix(self, tensor):
        if not torch.is_tensor(tensor):
            raise TypeError("Input must be a PyTorch tensor.")
        if tensor.ndim != 2 or tensor.shape[0] != tensor.shape[1]:
            return False
        identity = torch.eye(tensor.shape[0], device=tensor.device)
        return torch.all(torch.eq(tensor, identity))
    


class OFTInjectedLinear_with_norm(nn.Module):
    def __init__(
        self, in_features, out_features, bias=False, r=4, eps=1e-5, is_coft=True, block_share=False,
    ):
        super().__init__()

        assert in_features % r == 0, "in_features must be divisible by r"

        # Get the number of available GPUs
        self.num_gpus = torch.cuda.device_count()
        # Set the device IDs for distributed training
        self.device_ids = list(range(self.num_gpus))

        self.in_features=in_features
        self.out_features=out_features

        # Define the fixed Linear layer: v
        self.OFT = torch.nn.Linear(in_features=in_features, out_features=out_features, bias=bias)

        # Define the reduction rate:
        self.r = r
        self.is_coft = is_coft

        #self.filt_shape = [in_features, in_features]
        self.fix_filt_shape = [in_features, out_features]

        # Define the scaling factors
        self.scaling_factors = nn.Parameter(torch.ones(out_features, 1))

        # Define the trainable matrix parameter: R
        self.block_share=block_share
        if self.block_share:
            # Initialized as an identity matrix
            self.R_shape = [in_features // self.r, in_features // self.r]
            self.R = nn.Parameter(torch.zeros(self.R_shape[0], self.R_shape[0]), requires_grad=False)
  
            self.eps = eps * self.R_shape[0] * self.R_shape[0]
        else:
            # Initialized as an identity matrix
            self.R_shape = [self.r, in_features // self.r, in_features // self.r]
            R = torch.zeros(self.R_shape[1], self.R_shape[1])
            R = torch.stack([R] * self.r)
            self.R = nn.Parameter(R, requires_grad=False)
            self.eps = eps * self.R_shape[1] * self.R_shape[1]

    def forward(self, x):
        orig_dtype = x.dtype
        dtype = self.R.dtype

        if self.block_share:
            if self.is_coft:
                with torch.no_grad():
                    self.R.copy_(project(self.R, eps=self.eps))
            orth_rotate = self.cayley(self.R)
        else:
            if self.is_coft:
                with torch.no_grad():
                    self.R.copy_(project_batch(self.R, eps=self.eps))
            orth_rotate = self.cayley_batch(self.R)

        # Block-diagonal parametrization
        block_diagonal_matrix = self.block_diagonal(orth_rotate)

        # fix filter
        fix_filt = self.OFT.weight.data
        fix_filt = torch.transpose(fix_filt, 0, 1)
        filt = torch.mm(block_diagonal_matrix, fix_filt.to(dtype))
        filt = torch.transpose(filt, 0, 1)

        filt_scaled = filt * self.scaling_factors
 
        # Apply the trainable identity matrix
        bias_term = self.OFT.bias.data if self.OFT.bias is not None else None
        out = nn.functional.linear(input=x, weight=filt_scaled, bias=bias_term)
        # out = nn.functional.linear(input=x, weight=fix_filt.transpose(0, 1), bias=bias_term)

        return out #.to(orig_dtype)

    def cayley(self, data):
        r, c = list(data.shape)
        # Ensure the input matrix is skew-symmetric
        skew = 0.5 * (data - data.t())
        I = torch.eye(r, device=data.device)
        # Perform the Cayley parametrization
        Q = torch.mm(I + skew, torch.inverse(I - skew))
        # Q = torch.mm(I - skew, torch.inverse(I + skew))
        return Q
    
    def cayley_batch(self, data):
        b, r, c = data.shape
        # Ensure the input matrix is skew-symmetric
        skew = 0.5 * (data - data.transpose(1, 2))
        # I = torch.eye(r, device=data.device).unsqueeze(0).repeat(b, 1, 1)
        I = torch.eye(r, device=data.device).unsqueeze(0).expand(b, r, c)

        # Perform the Cayley parametrization
        Q = torch.bmm(I + skew, torch.inverse(I - skew))

        return Q

    def block_diagonal(self, R):
        if self.block_share:
            # Create a list of R repeated block_count times
            blocks = [R] * self.r
        else:
            # Create a list of R slices along the third dimension
            blocks = [R[i, ...] for i in range(self.r)]

        # Use torch.block_diag to create the block diagonal matrix
        A = torch.block_diag(*blocks)

        return A

    def is_orthogonal(self, R, eps=1e-5):
        with torch.no_grad():
            RtR = torch.matmul(R.t(), R)
            diff = torch.abs(RtR - torch.eye(R.shape[1], dtype=R.dtype, device=R.device))
            return torch.all(diff < eps)

    def is_identity_matrix(self, tensor):
        if not torch.is_tensor(tensor):
            raise TypeError("Input must be a PyTorch tensor.")
        if tensor.ndim != 2 or tensor.shape[0] != tensor.shape[1]:
            return False
        identity = torch.eye(tensor.shape[0], device=tensor.device)
        return torch.all(torch.eq(tensor, identity))



class OFTInjectedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, eps=1e-3, is_coft=True, block_share=False):
        super().__init__()

        self.in_channels=in_channels
        self.out_channels=out_channels
        self.kernel_size=kernel_size[0]
        self.stride=stride
        self.padding=padding
        self.bias=bias

        self.block_share=block_share
        self.is_coft=is_coft
 
        # Define the fixed Conv2d layer: v
        self.OFT = nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, stride=self.stride, padding=self.padding, bias=self.bias)

        self.filt_shape = [self.out_channels, self.in_channels, self.kernel_size, self.kernel_size]
        self.fix_filt_shape = [self.kernel_size * self.kernel_size * self.in_channels, self.out_channels]

        # Define the trainable matrix parameter: R
        if self.block_share:
            # Initialized as an identity matrix
            self.R_shape = [self.kernel_size * self.kernel_size, self.kernel_size * self.kernel_size]
            self.R = nn.Parameter(torch.zeros(self.R_shape[0], self.R_shape[0]), requires_grad=True)
  
            self.eps = eps * self.R_shape[0] * self.R_shape[0]
        else:
            # Initialized as an identity matrix
            self.R_shape = [self.in_channels, self.kernel_size * self.kernel_size, self.kernel_size * self.kernel_size]
            R = torch.zeros(self.R_shape[1], self.R_shape[1])
            R = torch.stack([R] * self.in_channels)
            self.R = nn.Parameter(R, requires_grad=True)

            self.eps = eps * self.R_shape[1] * self.R_shape[1]

    def forward(self, x):
        if self.block_share:
            with torch.no_grad():
                self.R.copy_(project(self.R, eps=self.eps))
            orth_rotate = self.cayley(self.R)
        else:
            with torch.no_grad():
                self.R.copy_(project_batch(self.R, eps=self.eps))
            orth_rotate = self.cayley_batch(self.R)

        # Block-diagonal parametrization
        block_diagonal_matrix = self.block_diagonal(orth_rotate)

        # fix filter
        fix_filt = self.OFT.weight.data
        fix_filt = fix_filt.view(self.fix_filt_shape)
        filt = torch.mm(block_diagonal_matrix, fix_filt)
        filt = filt.view(self.filt_shape)

        # Apply the trainable identity matrix
        bias_term = self.OFT.bias.data if self.OFT.bias is not None else None
        out = F.conv2d(input=x, weight=filt, bias=bias_term, stride=self.stride, padding=self.padding)
        
        return out 

    def cayley(self, data):
        r, c = list(data.shape)
        # Ensure the input matrix is skew-symmetric
        skew = 0.5 * (data - data.t())
        I = torch.eye(r, device=data.device)
        # Perform the Cayley parametrization
        Q = torch.mm(I + skew, torch.inverse(I - skew))
        # Q = torch.mm(I - skew, torch.inverse(I + skew))
        return Q
    
    def cayley_batch(self, data):
        b, r, c = data.shape
        # Ensure the input matrix is skew-symmetric
        skew = 0.5 * (data - data.transpose(1, 2))
        # I = torch.eye(r, device=data.device).unsqueeze(0).repeat(b, 1, 1)
        I = torch.eye(r, device=data.device).unsqueeze(0).expand(b, r, c)

        # Perform the Cayley parametrization
        Q = torch.bmm(I + skew, torch.inverse(I - skew))

        return Q

    def block_diagonal(self, R):
        if self.block_share:
            # Create a list of R repeated block_count times
            blocks = [R] * self.in_channels
        else:
            # Create a list of R slices along the third dimension
            blocks = [R[i, ...] for i in range(self.in_channels)]

        # Use torch.block_diag to create the block diagonal matrix
        A = torch.block_diag(*blocks)

        return A

    def is_orthogonal(self, R, eps=1e-5):
        with torch.no_grad():
            RtR = torch.matmul(R.t(), R)
            diff = torch.abs(RtR - torch.eye(R.shape[1], dtype=R.dtype, device=R.device))
            return torch.all(diff < eps)


UNET_DEFAULT_TARGET_REPLACE = {"CrossAttention", "Attention", "GEGLU"}

UNET_CONV_TARGET_REPLACE = {"ResBlock"}

UNET_EXTENDED_TARGET_REPLACE = {"ResBlock", "CrossAttention", "Attention", "GEGLU"}

TEXT_ENCODER_DEFAULT_TARGET_REPLACE = {"CLIPAttention"}

TEXT_ENCODER_EXTENDED_TARGET_REPLACE = {"CLIPAttention"}

DEFAULT_TARGET_REPLACE = UNET_DEFAULT_TARGET_REPLACE

EMBED_FLAG = "<embed>"


def _find_children(
    model,
    search_class: List[Type[nn.Module]] = [nn.Linear],
):
    """
    Find all modules of a certain class (or union of classes).
    Returns all matching modules, along with the parent of those moduless and the
    names they are referenced by.
    """
    result = []
    for parent in model.modules():
        for name, module in parent.named_children():
            if any([isinstance(module, _class) for _class in search_class]):
                result.append((parent, name, module))  # Append the result to the list

    return result  # Return the list instead of using 'yield'


def _find_modules_v2(
    model,
    ancestor_class: Optional[Set[str]] = None,
    search_class: List[Type[nn.Module]] = [nn.Linear],
    exclude_children_of: Optional[List[Type[nn.Module]]] = [
        OFTInjectedLinear,
        OFTInjectedConv2d,
    ],
):
    """
    Find all modules of a certain class (or union of classes) that are direct or
    indirect descendants of other modules of a certain class (or union of classes).
    Returns all matching modules, along with the parent of those moduless and the
    names they are referenced by.
    """

    # Get the targets we should replace all linears under
    if ancestor_class is not None:
        ancestors = (
            module
            for module in model.modules()
            if module.__class__.__name__ in ancestor_class
        )
    else:
        # this, incase you want to naively iterate over all modules.
        ancestors = [module for module in model.modules()]

    results = []
    # For each target find every linear_class module that isn't a child of a OFTInjectedLinear
    for ancestor in ancestors:
        for fullname, module in ancestor.named_modules():
            if any([isinstance(module, _class) for _class in search_class]):
                # Find the direct parent if this is a descendant, not a child, of target
                *path, name = fullname.split(".")
                parent = ancestor
                while path:
                    parent = parent.get_submodule(path.pop(0))
                # Skip this linear if it's a child of a OFTInjectedLinear
                if exclude_children_of and any(
                    [isinstance(parent, _class) for _class in exclude_children_of]
                ):
                    continue
                results.append((parent, name, module))  # Append the result to the list

    return results  # Return the list instead of using 'yield'

def _find_modules_old(
    model,
    ancestor_class: Set[str] = DEFAULT_TARGET_REPLACE,
    search_class: List[Type[nn.Module]] = [nn.Linear],
    exclude_children_of: Optional[List[Type[nn.Module]]] = [OFTInjectedLinear],
):
    ret = []
    for _module in model.modules():
        if _module.__class__.__name__ in ancestor_class:

            for name, _child_module in _module.named_modules():
                if _child_module.__class__ in search_class:
                    ret.append((_module, name, _child_module))

    return ret


_find_modules = _find_modules_v2
# _find_modules = _find_modules_old

def inject_trainable_oft(
    model: nn.Module,
    target_replace_module: Set[str] = DEFAULT_TARGET_REPLACE,
    verbose: bool = False,
    r: int = 4,
    eps: float = 1e-5,
    coft: bool = True,
    block_share: bool = False,
):
    """
    inject oft into model, and returns oft parameter groups.
    """

    require_grad_params = []
    names = []

    for _module, name, _child_module in _find_modules(
        model, target_replace_module, search_class=[nn.Linear]
    ):

        weight = _child_module.weight
        bias = _child_module.bias
        if verbose:
            print("OFT Injection : injecting oft into ", name)
            print("OFT Injection : weight shape", weight.shape)
        _tmp = OFTInjectedLinear(
            _child_module.in_features,
            _child_module.out_features,
            _child_module.bias is not None,
            r=r,
            eps=eps,
            is_coft=coft,
            block_share=block_share,
        )
        _tmp.OFT.weight = weight
        if bias is not None:
            _tmp.OFT.bias = bias

        # switch the module
        _tmp.to(_child_module.weight.device).to(_child_module.weight.dtype)
        _module._modules[name] = _tmp

        require_grad_params.append(_module._modules[name].R)

        _module._modules[name].R.requires_grad = True
        names.append(name)

    return require_grad_params, names


def inject_trainable_oft_with_norm(
    model: nn.Module,
    target_replace_module: Set[str] = DEFAULT_TARGET_REPLACE,
    verbose: bool = False,
    r: int = 4,
    eps: float = 1e-5,
    coft: bool = True,
    block_share: bool = False,
):
    """
    inject lora into model, and returns lora parameter groups.
    """

    require_grad_params = []
    names = []

    for _module, name, _child_module in _find_modules(
        model, target_replace_module, search_class=[nn.Linear]
    ):

        weight = _child_module.weight
        bias = _child_module.bias
        if verbose:
            print("OFT Injection : injecting oft into ", name)
            print("OFT Injection : weight shape", weight.shape)
        _tmp = OFTInjectedLinear_with_norm(
            _child_module.in_features,
            _child_module.out_features,
            _child_module.bias is not None
        )
        _tmp.OFT.weight = weight
        if bias is not None:
            _tmp.OFT.bias = bias

        # switch the module
        _tmp.to(_child_module.weight.device).to(_child_module.weight.dtype)
        _module._modules[name] = _tmp

        require_grad_params.append(_module._modules[name].scaling_factors)

        _module._modules[name].scaling_factors.requires_grad = True
        names.append(name)

    return require_grad_params, names



def inject_trainable_oft_extended(
    model: nn.Module,
    target_replace_module: Set[str] = UNET_EXTENDED_TARGET_REPLACE,
    verbose: bool = False,
    r: int = 4,
    eps: float = 1e-5,
    coft: bool = True,
    block_share: bool = False,
):
    """
    inject lora into model, and returns lora parameter groups.
    """

    require_grad_params = []
    names = []

    for _module, name, _child_module in _find_modules(
        model, target_replace_module, search_class=[nn.Linear, nn.Conv2d]
    ):
        if _child_module.__class__ == nn.Linear:
            weight = _child_module.weight
            bias = _child_module.bias
            _tmp = OFTInjectedLinear(
                _child_module.in_features,
                _child_module.out_features,
                _child_module.bias is not None,
            )
            _tmp.OFT.weight = weight
            if bias is not None:
                _tmp.OFT.bias = bias
        elif _child_module.__class__ == nn.Conv2d:
            weight = _child_module.weight
            bias = _child_module.bias
            _tmp = OFTInjectedConv2d(
                _child_module.in_channels,
                _child_module.out_channels,
                _child_module.kernel_size,
                _child_module.stride,
                _child_module.padding,
                #_child_module.dilation,
                #_child_module.groups,
                _child_module.bias is not None,
            )

            _tmp.OFT.weight = weight
            if bias is not None:
                _tmp.OFT.bias = bias

        # switch the module
        _tmp.to(_child_module.weight.device).to(_child_module.weight.dtype)
        if bias is not None:
            _tmp.to(_child_module.bias.device).to(_child_module.bias.dtype)

        _module._modules[name] = _tmp

        require_grad_params.append(_module._modules[name].R)

        _module._modules[name].R.requires_grad = True
        names.append(name)

    return require_grad_params, names


def inject_trainable_oft_conv(
    model: nn.Module,
    target_replace_module: Set[str] = UNET_CONV_TARGET_REPLACE,
    verbose: bool = False,
    r: int = 4,
    eps: float = 1e-5,
    coft: bool = True
):
    """
    inject lora into model, and returns lora parameter groups.
    """

    require_grad_params = []
    names = []

    for _module, name, _child_module in _find_modules(
        model, target_replace_module, search_class=[nn.Conv2d]
    ):
        if _child_module.__class__ == nn.Conv2d:
            weight = _child_module.weight
            bias = _child_module.bias
            _tmp = OFTInjectedConv2d(
                _child_module.in_channels,
                _child_module.out_channels,
                _child_module.kernel_size,
                _child_module.stride,
                _child_module.padding,
                # _child_module.dilation,
                # _child_module.groups,
                _child_module.bias is not None,
            )

            _tmp.OFT.weight = weight
            if bias is not None:
                _tmp.OFT.bias = bias

        # switch the module
        _tmp.to(_child_module.weight.device).to(_child_module.weight.dtype)
        if bias is not None:
            _tmp.to(_child_module.bias.device).to(_child_module.bias.dtype)

        _module._modules[name] = _tmp

        require_grad_params.append(_module._modules[name].R)

        _module._modules[name].R.requires_grad = True
        names.append(name)

    return require_grad_params, names

