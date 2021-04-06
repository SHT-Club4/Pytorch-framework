from torch.hub import load_state_dict_from_url
import torch.nn as nn
import torch
import torchvision
from models import alexnet
from models import resnet18, resnet34, resnet50, resnet101, resnet152
from models import densenet121, densenet169, densenet161, densenet201
from models import LOCAL_PRETRAINED, model_urls


def Alexnet(num_classes, test=False):
    model = alexnet()
    if not test:
        if LOCAL_PRETRAINED['alexnet'] == None:
            state_dict = load_state_dict_from_url(model_urls['alexnet'], progress=True)
        else:
            state_dict = torch.load(LOCAL_PRETRAINED['alexnet'])
        model.load_state_dict(state_dict)
    # 修改分类网络最后一层输出为类别数
    model.classifier.out_features = num_classes
    return model


def Resnet18(num_classes, test=False):
    model = resnet18()
    if not test:
        if LOCAL_PRETRAINED['resnet18'] == None:
            state_dict = load_state_dict_from_url(model_urls['resnet18'], progress=True)
        else:
            state_dict = torch.load(LOCAL_PRETRAINED['resnet18'])
        model.load_state_dict(state_dict)
    fc_features = model.fc.in_features
    model.fc = nn.Linear(fc_features, num_classes)
    return model


def Resnet34(num_classes, test=False):
    model = resnet34()
    if not test:
        if LOCAL_PRETRAINED['resnet34'] == None:
            state_dict = load_state_dict_from_url(model_urls['resnet34'], progress=True)
        else:
            state_dict = torch.load(LOCAL_PRETRAINED['resnet34'])
        model.load_state_dict(state_dict)
    fc_features = model.fc.in_features
    model.fc = nn.Linear(fc_features, num_classes)
    return model


def Resnet50(num_classes, test=False):
    model = resnet50()
    if not test:
        if LOCAL_PRETRAINED['resnet50'] == None:
            state_dict = load_state_dict_from_url(model_urls['resnet50'], progress=True)
        else:
            state_dict = torch.load(LOCAL_PRETRAINED['resnet50'])
        model.load_state_dict(state_dict)
    fc_features = model.fc.in_features
    model.fc = nn.Linear(fc_features, num_classes)
    return model


def Resnet101(num_classes, test=False):
    model = resnet101()
    if not test:
        if LOCAL_PRETRAINED['resnet101'] == None:
            state_dict = load_state_dict_from_url(model_urls['resnet101'], progress=True)
        else:
            state_dict = torch.load(LOCAL_PRETRAINED['resnet101'])
        model.load_state_dict(state_dict)
    fc_features = model.fc.in_features
    model.fc = nn.Linear(fc_features, num_classes)
    return model


def Resnet152(num_classes, test=False):
    model = resnet152()
    if not test:
        if LOCAL_PRETRAINED['resnet152'] == None:
            state_dict = load_state_dict_from_url(model_urls['resnet152'], progress=True)
        else:
            state_dict = torch.load(LOCAL_PRETRAINED['resnet152'])
        model.load_state_dict(state_dict)
    fc_features = model.fc.in_features
    model.fc = nn.Linear(fc_features, num_classes)
    return model


def Densenet121(num_classes, test=False):
    model = densenet121()
    if not test:
        if LOCAL_PRETRAINED['densenet121'] == None:
            state_dict = load_state_dict_from_url(model_urls['densenet121'], progress=True)
        else:
            state_dict = torch.load(LOCAL_PRETRAINED['densenet121'])
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.split('.')[0] == 'features' and (len(k.split('.'))) > 4:
                k = k.split('.')[0] + '.' + k.split('.')[1] + '.' + k.split('.')[2] + '.' + k.split('.')[-3] + k.split('.')[-2] + '.' + k.split('.')[-1]
            else:
                pass
            new_state_dict[k] = v
        model.load_state_dict(new_state_dict)
    fc_features = model.classifier.infeatures
    model.classifier = nn.Linear(fc_features, num_classes)
    return model


def Densenet161(num_classes, test=False):
    model = densenet161()
    if not test:
        if LOCAL_PRETRAINED['densenet161'] == None:
            state_dict = load_state_dict_from_url(model_urls['densenet161'], progress=True)
        else:
            state_dict = torch.load(LOCAL_PRETRAINED['densenet161'])
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.split('.')[0] == 'features' and (len(k.split('.'))) > 4:
                k = k.split('.')[0] + '.' + k.split('.')[1] + '.' + k.split('.')[2] + '.' + k.split('.')[-3] + k.split('.')[-2] + '.' + k.split('.')[-1]
            else:
                pass
            new_state_dict[k] = v
        model.load_state_dict(new_state_dict)
    fc_features = model.classifier.infeatures
    model.classifier = nn.Linear(fc_features, num_classes)
    return model


def Densenet169(num_classes, test=False):
    model = densenet169()
    if not test:
        if LOCAL_PRETRAINED['densenet169'] == None:
            state_dict = load_state_dict_from_url(model_urls['densenet169'], progress=True)
        else:
            state_dict = torch.load(LOCAL_PRETRAINED['densenet169'])
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.split('.')[0] == 'features' and (len(k.split('.'))) > 4:
                k = k.split('.')[0] + '.' + k.split('.')[1] + '.' + k.split('.')[2] + '.' + k.split('.')[-3] + k.split('.')[-2] + '.' + k.split('.')[-1]
            else:
                pass
            new_state_dict[k] = v
        model.load_state_dict(new_state_dict)
    fc_features = model.classifier.infeatures
    model.classifier = nn.Linear(fc_features, num_classes)
    return model


def Densenet201(num_classes, test=False):
    model = densenet201()
    if not test:
        if LOCAL_PRETRAINED['densenet201'] == None:
            state_dict = load_state_dict_from_url(model_urls['densenet201'], progress=True)
        else:
            state_dict = torch.load(LOCAL_PRETRAINED['densenet201'])
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.split('.')[0] == 'features' and (len(k.split('.'))) > 4:
                k = k.split('.')[0] + '.' + k.split('.')[1] + '.' + k.split('.')[2] + '.' + k.split('.')[-3] + k.split('.')[-2] + '.' + k.split('.')[-1]
            else:
                pass
            new_state_dict[k] = v
        model.load_state_dict(new_state_dict)
    fc_features = model.classifier.infeatures
    model.classifier = nn.Linear(fc_features, num_classes)
    return model
