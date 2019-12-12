 
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

# A simple hook class that returns the input and output of a layer during forward/backward pass
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
#######################################################################################
TORCHVISION_MODEL_NAMES = sorted(
                            name for name in models.__dict__
                            if name.islower() and not name.startswith("__")
                            and callable(models.__dict__[name]))

model = models.resnet50(pretrained=True)
#model = getattr(models, 'mobilenet_v2')(pretrained=True)
#model = EfficientNet.from_pretrained('efficientnet-b7')
#################################################################################3
# register hooks on each layer
hookF = [Hook(layer) for layer in list(model.modules())]


im_object =Image.open("/home/maziar/WA/exampleofhooks/cat_224.jpg")
normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
to_tensor = transforms.ToTensor()
scaler = transforms.Scale((224, 224))
dataa = normalize(to_tensor(scaler(im_object))).unsqueeze(0)


out = model(dataa)

#out.backward(torch.tensor([1,1],dtype=torch.float),retain_graph=True)
#! loss.backward(retain_graph=True)  # doesn't work with backward hooks, 
#! since it's not a network layer but an aggregated result from the outputs of last layer vs target 

print('***'*3+'  Forward Hooks Inputs & Outputs  '+'***'*3)
tot_num_mac = 0
tot_num_add = 0
tot_num_mult = 0
tot_num_comp = 0

for hook in hookF:
    print(hook.num_mac)
    tot_num_mac = tot_num_mac + hook.num_mac
    tot_num_add += hook.num_add
    tot_num_mult += hook.num_mult
    tot_num_comp += hook.num_comp
    
    print('---'*17)
    print('\n')
print(f"total num of mac is {tot_num_mac}")
print(f"total num of add is {tot_num_add}")
print(f"total num of multiplcation is {tot_num_mult}")
print(f"total num of comparison in Maxpool is {tot_num_comp}")
#print(f"total num of add in AvgPool is {tot_num_add_avg}")

# print('***'*3+'  Backward Hooks Inputs & Outputs  '+'***'*3)
# for hook in hookB:             
#     print(hook.input)          
# #     print(hook.output)         
# #     print('---'*17)
