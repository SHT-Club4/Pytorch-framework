# pytorch-framework
利用pytorch实现图像分类，其中包含alexnet，resnet，densenet（其他网络可添加）等图像分类网络

## 功能
* 基础功能：使用pytorch实现图像分类
* 带有warmup的step学习率调整
* 带有warmup的cosine学习率调整

## 运行环境
* python3.7
* pytorch 1.7

## 代码仓库使用

### 1.准备数据集
原始数据集形式为，同一个类别的图像存储在同一个文件夹下，所有图像存储在主文件夹data下

```text
# 验证集由train文件夹中分离出来
|--data
    |--train
        |--label1
            |--*.jpg
        |--label2
            |--*.jpg
        |--label3
            |--*.jpg
        ...
# 测试集不用分类别，供最后predict.py测试和
    |--test
        |--*.jpg
        |--*.jpg
        |--*.jpg
        ...
```
利用preprocess.py将数据集格式进行转换（个人习惯）
```shell
python ./data/preprocess.py
```
转换后的数据集为，将训练集的路径与类别存储在train.txt文件中，测试集存储在val.txt中

其中txt文件中的内容为
```text
# train.txt
/home/gong/sht/data/train/label1/*.jpg label
...

# val.txt
/home/gong/sht/data/train/label1/*.jpg
...
```

运行preprocess.py后，data文件夹全貌为
```text
|-- data
    |--train
        |--label1
            |--*.jpg
        |--label2
            |--*.jpg
        |--label    
            |--*.jpg
        ...
    |--val
        |--label1
            |--*.jpg
        |--label2
            |--*.jpg
        |--label    
            |--*.jpg
        ...
    |--test
        |--*.jpg
        |--*.jpg
        |--*.jpg
        ...
    |--train.txt
    |--val.txt
    |--test.txt
```

### 2.训练
在`cfg.py`中修改合适的参数，参数具体含义已将注释文件中

在`train.py`中选择合适的模型

```shell
pyhton train.py
```

### 3.预测
在`cfg.py`中的`TRAINED_MODEL`设置已训练好的权重文件的位置

```shell
python predict.py
```

# 一些实现细节
## 增加新模型的步骤（以alexnet为例）
### 1. 在`models/vision`中创建*.py（alexnet.py)
```python
import torch.nn as nn
from .utils import load_state_dict_from_url

__all__ = ['AlexNet', 'alexnet']

model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}

class AlexNet(nn.Module):
    pass

def alexnet(pretrained=False, progress=True, **kwargs):
    model = AlexNet(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['alexnet'], progress=progress)
        model.load_state_dict(state_dict)
    return model
```
其中`Alexnet`为自定义模型，上例为代码最后返回模型的格式
### 2. 在`models/vision/__init__.py`中导入*.py
```
from .alexnet import *
```
### 3. 在`models/__init__.py`中选择是否存在预训练模型和预训练模型的下载地址
```python
LOCAL_PRETRAINED = {
    'alexnet': None
}

model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth'
}
```
`LOCAL_PRETRAINED`中如果填None，则默认从`model_urls`中对应模型的地址中下载预训练模型

`LOCAL_PRETRAINED`中如果有地址，则从该地址中加载预训练模型
### 4. 在`build_model.py`中进行模型的实例化
```python
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
```
### 5. 在`cfg.py`中导入模型文件
```python
from models import Alexnet
```
并修改`cfg.py`中对应参数值

在`MODEL_NAMES`中增加相应键值对
### 6. 准备结束，可以开始训练

---

# TODO LIST
-[ ] 增加分类网络模型
    -[ ] googleNet
    -[ ] inceptionNet
    -[ ] mobileNet
    -[ ] VGG
    -[ ] shuffleNetv2
    -[ ] squeezeNet
    -[ ] resNext
    -[ ] efficientNet
    
-[ ] 特征图可视化
-[ ] C++部署