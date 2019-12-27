
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



#get all the pre trained model from torch
TORCHVISION_MODEL_NAMES = sorted(
                            name for name in models.__dict__
                            if name.islower() and not name.startswith("__")
                            and callable(models.__dict__[name]))


#select our desired model
model = models.resnet18(pretrained=True)
#model = getattr(models, 'mobilenet_v2')(pretrained=True)
#model = EfficientNet.from_pretrained('efficientnet-b7')

#####

#create a dictionary of module names
hookF = {}
for name, module in model.named_modules():
    hookF[name] = Hook(module)

#define a function to read an image and change it to a tensor
def data_loader(path):
    im_object =Image.open(path)
    normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
    to_tensor = transforms.ToTensor()
    scaler = transforms.Scale((224, 224))
    #scaler = transforms.Resize((224, 224))
    data = normalize(to_tensor(scaler(im_object))).unsqueeze(0)
    return data

#feed data to our model
path = " a path to your photo"
out = model(data_loader(path))


#define a function to seperate every 3 digits by a comma
def group_digit(number):
    s = '%d' % number
    groups = []
    while s and s[-1].isdigit():
        groups.append(s[-3:])
        s = s[:-3]
    return s + ','.join(reversed(groups))

print('***'*3+'  Forward Hooks Inputs & Outputs  '+'***'*3)
tot_num_mac = 0
tot_num_add = 0
tot_num_mult = 0
tot_num_comp = 0

for key, value in hookF.items():
    print(f'layer name: {key}')
    print(f'input size: {value.input_size}')
    print(f'kernel size: {value.kernel_size}')
    print(f'padding: {value.padding_size}')
    print(f"output size: {value.output_size}")
    print(f'sparsity: {value.percentage}%')
    print(f'Number of ops for this layer: {group_digit(value.num_mac)}')
    tot_num_mac = tot_num_mac + value.num_mac
    tot_num_add += value.num_add
    tot_num_mult += value.num_mult
    tot_num_comp += value.num_comp
    print('---'*17)
    print('\n')
    
    print('---'*17)
    print('\n')
print(f"total num of mac is {group_digit(tot_num_mac)}")
print(f"total num of add is {group_digit(tot_num_add)}")
print(f"total num of multiplcation is {group_digit(tot_num_mult)}")
print(f"total num of comparison in Maxpool is {group_digit(tot_num_comp)}")
#print(f"total num of add in AvgPool is {tot_num_add_avg}")

