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



class Hook():
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.num_mac = 0
        self.num_mult= 0
        self.num_add = 0
        self.num_comp = 0
        self.num_mac_avg = 0
        
    def hook_fn(self, module, input, output):


        if isinstance(module, nn.Conv2d):
            output_size = output.shape[0] * output.shape[1] * output.shape[2] * output.shape[3]
            num_ops_per_one_output = (module.in_channels / module.groups) * module.kernel_size[1] * module.kernel_size[0]
            self.num_mac = num_ops_per_one_output * output_size
            self.num_mult = int(self.num_mac)
            self.num_add = int(self.num_mac)


        if isinstance(module, nn.MaxPool2d):
            output_size = output.shape[0] * output.shape[1] * output.shape[2] * output.shape[3]
            num_compar_per_output_max =  module.kernel_size * module.kernel_size
            self.num_comp = num_compar_per_output_max * output_size

        if isinstance(module, nn.AdaptiveAvgPool2d):
            output_size = output.shape[0] * output.shape[1] * output.shape[2] * output.shape[3]
            num_compar_per_output_max =  input[0].shape[2] * input[0].shape[3]
            self.num_add= num_compar_per_output_max * output_size
            self.num_mult = 1 * output_size


            
    def close(self):
        self.hook.remove()