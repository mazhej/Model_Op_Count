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

# A simple hook class that returns the input and output of a layer during forward/backward pass
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
#######################################################################################
model = models.resnet50(pretrained=True)
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
tot_num_ops = 0
for hook in hookF:
    print(hook.num_ops)
    tot_num_ops = tot_num_ops + hook.num_ops
    print('---'*17)
    print('\n')
print(tot_num_ops)
