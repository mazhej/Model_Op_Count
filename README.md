# Introduction
Neural Networks have become very popular in the last decade. Computer Vision, Natural Language Processing, and all other areas of AI are growing so fast.
In Deep Neural Networks, one of the challenges we face is computational cost. Recent CV models such as [MobileNet](https://pytorch.org/hub/pytorch_vision_mobilenet_v2/) and [Resnet](https://pytorch.org/hub/pytorch_vision_resnet/) are so deep and have many layers.
There are many ways to reduce the the complexity of these models, one of which is Pruning. To do so, it is realy important for us to know which layer should be pruned.

This repository help us calculate the number of operations (MAC, multliplication + accumulation) in different layers such as Conv, MaxPool, and AvgPool.

### Prerequisites
This project has been implemented in [Pytorch](https://pytorch.org/) using GPU. To learn how to install Pytorch refer to this [LINK](https://pytorch.org/get-started/locally/)

### Installing 
```
git clone https://github.com/mazhej/Model_Op_Count.git
```
The main part of this project used Hooks technique to register input and output. To learn more about how to register a hook in pytorch you can refer to this [LINK](https://pytorch.org/tutorials/beginner/former_torchies/nnft_tutorial.html?highlight=hooks)

