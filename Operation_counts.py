 
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

#getting all pretrained models from torch
TORCHVISION_MODEL_NAMES = sorted(
                            name for name in models.__dict__
                            if name.islower() and not name.startswith("__")
                            and callable(models.__dict__[name]))

#choosing our desired pre trainded model as backbone
model = models.resnet50(pretrained=True)
#model = getattr(models, 'mobilenet_v2')(pretrained=True)
#model = EfficientNet.from_pretrained('efficientnet-b7')

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

