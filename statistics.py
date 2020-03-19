from PIL import *
from PIL import Image

import os

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
from statistics_hook import *



#get all the pre trained model from torch
TORCHVISION_MODEL_NAMES = sorted(
                            name for name in models.__dict__
                            if name.islower() and not name.startswith("__")
                            and callable(models.__dict__[name]))


#select our desired model
model = models.resnet50(pretrained=True)
#model = getattr(models, 'mobilenet_v2')(pretrained=True)
#model = EfficientNet.from_pretrained('efficientnet-b7')

#####

# register hooks on each layer
hookF = {}
for name, module in model.named_modules():
    hookF[name] = statistics(module)

#define a function to read an image and change it tensor
def data_loader(path="/home/maziar/WA/exampleofhooks/5.jpg"):
    im_object =Image.open(path)
    normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
    to_tensor = transforms.ToTensor()
    scaler = transforms.Scale((224, 224))
    #scaler = transforms.Resize((224, 224))
    data = normalize(to_tensor(scaler(im_object))).unsqueeze(0)
    return data

#feed our data to our model
out = model(data_loader())

#define a function to seperate every 3 digits by a comma
def group(number):
    s = '%d' % number
    groups = []
    while s and s[-1].isdigit():
        groups.append(s[-3:])
        s = s[:-3]
    return s + ','.join(reversed(groups))

def Average(lst): 
    return sum(lst) / len(lst)

print('***'*3+'  Forward Hooks Inputs & Outputs  '+'***'*3)

min_input_list = []
max_input_list = []


for key, value in hookF.items():
    print(f'layer name: {key}')
    print(f'min_input:{value.min_input}',f'max_input:{value.max_input}')
    print(f'min_output:{value.min_output}',f'max_output:{value.max_output}')
    
    min_input_list.append(value.min_input.detach().numpy())
    max_input_list.append(value.max_input.detach().numpy())
    print('---'*17)
    print('\n')
    
    print('---'*17)
    print('\n')

print(f'the avg_min: {Average(min_input_list)}')
print(f'the avg_max: {Average(max_input_list)}')
