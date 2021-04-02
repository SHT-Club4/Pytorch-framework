import os
home = os.path.expanduser('~')

# 预训练模型的存放位置
LOCAL_PRETRAINED = {
    'alexnet': None,

    'resnet18': home + '/weights/resnet18.pth',
    'resnet34': home + '/weights/resnet34.pth',
    'resnet50': home + '/weights/resnet50.pth',
    'resnet101': home + '/weights/resnet101.pth',
    'resnet152': home + '/weights/resnet152.pth',

    'densenet121': home + '/weights/densenet121.pth',
    'densenet161': home + '/weights/densenet161.pth',
    'densenet169': home + '/weights/densenet169.pth',
    'densenet201': home + '/weights/densenet201.pth',
}

model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',

    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',

    'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
    'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',
    'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
    'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
}

from .vision import *
from .build_model import *
