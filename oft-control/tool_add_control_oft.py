"""
This script utilizes code from ControlNet available at: 
https://github.com/lllyasviel/ControlNet/blob/main/tool_add_control.py

Original Author: Lvmin Zhang
License: Apache License 2.0
"""

import sys
import os
os.environ['HF_HOME'] = '/tmp'

# assert len(sys.argv) == 3, 'Args are wrong.'

# input_path = sys.argv[1]
# output_path = sys.argv[2]

import torch
from oldm.hack import disable_verbosity
disable_verbosity()
from oldm.model import create_model
from oft import inject_trainable_oft, inject_trainable_oft_conv, inject_trainable_oft_extended

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--input_path', type=str, default='./models/v1-5-pruned.ckpt')
parser.add_argument('--output_path', type=str, default='./models/control_sd15_ini_oft.ckpt')
parser.add_argument('--r', type=int, default=4)
parser.add_argument('--eps', type=float, default=1e-5)
parser.add_argument('--coft', action="store_true")
parser.add_argument('--block_share', action="store_true", default=False)

args = parser.parse_args()

assert os.path.exists(args.input_path), 'Input model does not exist.'
# assert not os.path.exists(output_path), 'Output filename already exists.'
assert os.path.exists(os.path.dirname(args.output_path)), 'Output path is not valid.'


def get_node_name(name, parent_name):
    if len(name) <= len(parent_name):
        return False, ''
    p = name[:len(parent_name)]
    if p != parent_name:
        return False, ''
    return True, name[len(parent_name):]


model = create_model(config_path='./configs/oft_ldm_v15.yaml')
model.model.requires_grad_(False)

unet_lora_params, train_names = inject_trainable_oft(model.model, r=args.r, eps=args.eps, is_coft=args.coft, block_share=args.block_share)
# unet_lora_params, train_names = inject_trainable_oft_conv(model.model, r=args.r, eps=args.eps, is_coft=args.coft, block_share=args.block_share)
# unet_lora_params, train_names = inject_trainable_oft_extended(model.model, r=args.r, eps=args.eps, is_coft=args.coft, block_share=args.block_share)

pretrained_weights = torch.load(args.input_path)
if 'state_dict' in pretrained_weights:
    pretrained_weights = pretrained_weights['state_dict']

scratch_dict = model.state_dict()

target_dict = {}
names = []
for k in scratch_dict.keys():
    names.append(k)

    if k in pretrained_weights:
        target_dict[k] = pretrained_weights[k].clone()
    else:
        if 'OFT.' in k:
            copy_k = k.replace('OFT.', '')
            target_dict[k] = pretrained_weights[copy_k].clone()
        else:
            target_dict[k] = scratch_dict[k].clone()
            print(f'These weights are newly added: {k}')

with open('model_names.txt', 'w') as file:
    for element in names:
        file.write(element + '\n')

model.load_state_dict(target_dict, strict=True)
torch.save(model.state_dict(), args.output_path)
print('Done.')
