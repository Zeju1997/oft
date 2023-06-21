"""
This script utilizes code from ControlNet available at: 
https://github.com/lllyasviel/ControlNet/blob/main/tool_add_control.py

Original Author: Lvmin Zhang
License: Apache License 2.0
"""

import sys
import os
os.environ['HF_HOME'] = '/tmp'

assert len(sys.argv) == 3, 'Args are wrong.'

input_path = sys.argv[1]
output_path = sys.argv[2]

assert os.path.exists(input_path), 'Input model does not exist.'
assert not os.path.exists(output_path), 'Output filename already exists.'
assert os.path.exists(os.path.dirname(output_path)), 'Output path is not valid.'

import torch
from share import *
from oldm.model import create_model
from oft import inject_trainable_oft, inject_trainable_oft_conv, inject_trainable_oft_extended

def get_node_name(name, parent_name):
    if len(name) <= len(parent_name):
        return False, ''
    p = name[:len(parent_name)]
    if p != parent_name:
        return False, ''
    return True, name[len(parent_name):]


model = create_model(config_path='./configs/oft_ldm_v15.yaml')
model.model.requires_grad_(False)
unet_lora_params, train_names = inject_trainable_oft(model.model)
# unet_lora_params, train_names = inject_trainable_oft_extended(model.model)
# unet_lora_params, train_names = inject_trainable_oft_conv(model.model)

pretrained_weights = torch.load(input_path)
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
torch.save(model.state_dict(), output_path)
print('Done.')
