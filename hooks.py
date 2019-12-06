#importing libraries
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

# A simple hook class that returns the input and output of a layer during forward pass
class Hook():
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.num_ops = 0
    def hook_fn(self, module, input, output):
        if isinstance(module, nn.Conv2d):
            num_ops_per_one_output = module.in_channels * module.kernel_size[1] * module.kernel_size[0]
            output_size = output.shape[0] * output.shape[1] * output.shape[2] * output.shape[3]
            self.num_ops = num_ops_per_one_output * output_size
            
    def close(self):
        self.hook.remove()

#########################
#Selecting resnet18 as our model
model = models.resnet18(pretrained=True)

#########################
# register hooks on each layer
hookF = [Hook(layer) for layer in list(model.modules())]

#Creat a function to load an image and change it to tensor.
#later on, this data will be passed to our model
def data_loader():
    im_object =Image.open("/home/maziar/WA/exampleofhooks/cat_224.jpg")
    normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
    to_tensor = transforms.ToTensor()
    scaler = transforms.Scale((224, 224))
    data= normalize(to_tensor(scaler(im_object))).unsqueeze(0)
    return data

#get the output of the model and save it
out = model(data_loader())

 
#print number of operations
print('***'*3+'  Forward Hooks Inputs & Outputs  '+'***'*3)
tot_num_ops = 0
for hook in hookF:
    print(hook.num_ops)
    tot_num_ops = tot_num_ops + hook.num_ops
    print('---'*17)
    print('\n')
print(f'the total number of conv operations: {tot_num_ops}')
