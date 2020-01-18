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



class statistics():
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.num_mac = 0
        self.num_mult= 0
        self.num_add = 0
        self.num_comp = 0
        self.num_mac_avg = 0
        self.input_size = []
        self.output_size = []
        self.kernel_size = []
        self.padding_size = []
        self.zero = 0
        self.percentage = 0
        self.min_input = 0
        self.max_input = 0
        self.min_output = 0
        self.max_output = 0
        
        
        
        
        

    def hook_fn(self, module, input, output):

        self.input_size = input[0].shape
        

        if isinstance(module, nn.Conv2d):
            self.output_size = output.shape
            self.kernel_size = module.kernel_size
            self.padding_size = module.padding
            output_size = output.shape[0] * output.shape[1] * output.shape[2] * output.shape[3]
            num_ops_per_one_output = (module.in_channels / module.groups) * module.kernel_size[1] * module.kernel_size[0]
            self.num_mac = num_ops_per_one_output * output_size
            self.num_mult = int(self.num_mac)
            self.num_add = int(self.num_mac)
            
            for param in input:
                if param is not None:
                    self.zero += param.numel() - param.nonzero().size(0)
            self.percentage =  (self.zero / (input[0].shape[2] * input[0].shape[3] * input[0].shape[1] * input[0].shape[0])) *100
            


        if isinstance(module, nn.MaxPool2d):
            self.output_size = output.shape
            self.kernel_size = module.kernel_size
            self.padding_size = module.padding
            output_size = output.shape[0] * output.shape[1] * output.shape[2] * output.shape[3]
            num_compar_per_output_max =  module.kernel_size * module.kernel_size
            self.num_comp = num_compar_per_output_max * output_size

        if isinstance(module, nn.AdaptiveAvgPool2d):
            self.output_size = output.shape
            output_size = output.shape[0] * output.shape[1] * output.shape[2] * output.shape[3]
            num_compar_per_output_max =  input[0].shape[2] * input[0].shape[3]
            self.num_add= num_compar_per_output_max * output_size
            self.num_mult = 1 * output_size

        self.min_input = torch.min(input[0])
        self.max_input = torch.max(input[0])
        self.min_output = torch.min(output)
        self.max_output = torch.max(output)
        #self.min_input_list.append(self.min_input.detach().numpy())

        print("")


            
    def close(self):
        self.hook.remove()