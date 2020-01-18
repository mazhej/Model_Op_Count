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
        self.min_input = 0
        self.max_input = 0
        self.min_output = 0
        self.max_output = 0
        
    def hook_fn(self, module, input, output):
        self.min_input = torch.min(input[0])
        self.max_input = torch.max(input[0])
        self.min_output = torch.min(output)
        self.max_output = torch.max(output)
        #self.min_input_list.append(self.min_input.detach().numpy())
    
    def close(self):
        self.hook.remove()