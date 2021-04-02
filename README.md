# pytorch-framework
利用pytorch实现图像分类，其中包含alexnet，resnet，densenet（其他网络可添加）等图像分类网络

## 实现功能
* 基础功能：使用pytorch实现图像分类
* 带有warmup的step学习率调整
* 带有warmup的cosine学习率调整

## 运行环境
* python3.7
* pytorch 1.7

## 代码仓库使用

### 数据集形式
原始数据集形式为，同一个类别的图像存储在同一个文件夹下，所有图像存储在主文件夹data下

```
|--data
    |--train
        |--label1
            |--*.jpg
        |--label2
            |--*.jpg
        |--label3
            |--*.jpg
        ...
    |--val
        |--label1
            |--*.jpg
        |--label2
            |--*.jpg
        |--label3
            |--*.jpg
        ...
```
利用preprocess.py将数据集格式进行转换（个人习惯）
```
python ./data/preprocess.py
```
