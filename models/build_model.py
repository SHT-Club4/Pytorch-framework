from torch.hub import load_state_dict_from_url
import torch.nn as nn
import torch
import torchvision
from models import vgg11, vgg13, vgg16, vgg19, vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn
from models import alexnet
from models import resnet18, resnet34, resnet50, resnet101, resnet152
from models import densenet121, densenet169, densenet161, densenet201
from models import LOCAL_PRETRAINED, model_urls


def VGG11(num_classes, test=False):
    model = vgg11()
    if not test:
        if LOCAL_PRETRAINED['vgg11'] is None:
            state_dict = load_state_dict_from_url(model_urls['vgg11'], progress=True)
        else:
            state_dict = torch.load(LOCAL_PRETRAINED['vgg11'])
        model.load_state_dict(state_dict)
    model.classifier.out_features = num_classes
    return model


def VGG13(num_classes, test=False):
    model = vgg13()
    if not test:
        if LOCAL_PRETRAINED['vgg13'] is None:
            state_dict = load_state_dict_from_url(model_urls['vgg13'], progress=True)
        else:
            state_dict = torch.load(LOCAL_PRETRAINED['vgg13'])
        model.load_state_dict(state_dict)
    model.classifier.out_features = num_classes
    return model


def VGG16(num_classes, test=False):
    model = vgg16()
    if not test:
        if LOCAL_PRETRAINED['vgg16'] is None:
            state_dict = load_state_dict_from_url(model_urls['vgg16'], progress=True)
        else:
            state_dict = torch.load(LOCAL_PRETRAINED['vgg16'])
        model.load_state_dict(state_dict)
    model.classifier.out_features = num_classes
    return model


def VGG19(num_classes, test=False):
    model = vgg19()
    if not test:
        if LOCAL_PRETRAINED['vgg19'] is None:
            state_dict = load_state_dict_from_url(model_urls['vgg19'], progress=True)
        else:
            state_dict = torch.load(LOCAL_PRETRAINED['vgg19'])
        model.load_state_dict(state_dict)
    model.classifier.out_features = num_classes
    return model


def VGG11_bn(num_classes, test=False):
    model = vgg11_bn()
    if not test:
        if LOCAL_PRETRAINED['vgg11_bn'] is None:
            state_dict = load_state_dict_from_url(model_urls['vgg11_bn'], progress=True)
        else:
            state_dict = torch.load(LOCAL_PRETRAINED['vgg11_bn'])
        model.load_state_dict(state_dict)
    model.classifier.out_features = num_classes
    return model


def VGG13_bn(num_classes, test=False):
    model = vgg13_bn()
    if not test:
        if LOCAL_PRETRAINED['vgg13_bn'] is None:
            state_dict = load_state_dict_from_url(model_urls['vgg13_bn'], progress=True)
        else:
            state_dict = torch.load(LOCAL_PRETRAINED['vgg13_bn'])
        model.load_state_dict(state_dict)
    model.classifier.out_features = num_classes
    return model


def VGG16_bn(num_classes, test=False):
    model = vgg16_bn()
    if not test:
        if LOCAL_PRETRAINED['vgg16_bn'] is None:
            state_dict = load_state_dict_from_url(model_urls['vgg16_bn'], progress=True)
        else:
            state_dict = torch.load(LOCAL_PRETRAINED['vgg16_bn'])
        model.load_state_dict(state_dict)
    model.classifier.out_features = num_classes
    return model


def VGG19_bn(num_classes, test=False):
    model = vgg19_bn()
    if not test:
        if LOCAL_PRETRAINED['vgg19_bn'] is None:
            state_dict = load_state_dict_from_url(model_urls['vgg19_bn'], progress=True)
        else:
            state_dict = torch.load(LOCAL_PRETRAINED['vgg19_bn'])
        model.load_state_dict(state_dict)
    model.classifier.out_features = num_classes
    return model


def Alexnet(num_classes, test=False):
    model = alexnet()
    if not test:
        if LOCAL_PRETRAINED['alexnet'] is None:
            state_dict = load_state_dict_from_url(model_urls['alexnet'], progress=True)
        else:
            state_dict = torch.load(LOCAL_PRETRAINED['alexnet'])
        model.load_state_dict(state_dict)
    model.classifier.out_features = num_classes
    return model


def Resnet18(num_classes, test=False):
    model = resnet18()
    if not test:
        if LOCAL_PRETRAINED['resnet18'] is None:
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
        if LOCAL_PRETRAINED['resnet34'] is None:
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
        if LOCAL_PRETRAINED['resnet50'] is None:
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
        if LOCAL_PRETRAINED['resnet101'] is None:
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
        if LOCAL_PRETRAINED['resnet152'] is None:
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
        if LOCAL_PRETRAINED['densenet121'] is None:
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
        if LOCAL_PRETRAINED['densenet161'] is None:
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
        if LOCAL_PRETRAINED['densenet169'] is None:
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
        if LOCAL_PRETRAINED['densenet201'] is None:
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
