 #https://www.kaggle.com/sironghuang/understanding-pytorch-hooks
import requests
from PIL import *
from PIL import Image
import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from efficientnet_pytorch import EfficientNet
from Hook import *


#get all the pre trained model from torch
TORCHVISION_MODEL_NAMES = sorted(
                            name for name in models.__dict__
                            if name.islower() and not name.startswith("__")
                            and callable(models.__dict__[name]))


#select our desired model

precision = 'fp32'
model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd', model_math=precision)
utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd_processing_utils')
model.to('cuda')
model.eval()

uris = ['http://images.cocodataset.org/val2017/000000397133.jpg']
inputs = [utils.prepare_input(uri) for uri in uris]
tensor = utils.prepare_tensor(inputs, precision == 'fp16')
#####

# register hooks on each layer
hookF = {}
for name, module in model.named_modules():
    hookF[name] = Hook(module)
    
#feed our data to our model
out = model(tensor)

print('***'*3+'  Forward Hooks Inputs & Outputs  '+'***'*3)
tot_num_mac = 0
tot_num_add = 0
tot_num_mult = 0
tot_num_comp = 0

def group(number):
    s = '%d' % number
    groups = []
    while s and s[-1].isdigit():
        groups.append(s[-3:])
        s = s[:-3]
    return s + ','.join(reversed(groups))


for key, value in hookF.items():
    print(f'layer name: {key}')
    print(f'input size: {value.input_size}')
    print(f'kernel size: {value.kernel_size}')
    print(f'padding: {value.padding_size}')
    print(f"output size: {value.output_size}")
    print(f'sparsity: {value.percentage}%')
    print(f'Number of ops for this layer: {group(value.num_mac)}')
    tot_num_mac = tot_num_mac + value.num_mac
    tot_num_add += value.num_add
    tot_num_mult += value.num_mult
    tot_num_comp += value.num_comp
    print('---'*17)
    print('\n')
    
    print('---'*17)
    print('\n')
print(f"total num of mac is {group(tot_num_mac)}")
print(f"total num of add is {group(tot_num_add)}")
print(f"total num of multiplcation is {group(tot_num_mult)}")
print(f"total num of comparison in Maxpool is {group(tot_num_comp)}")
