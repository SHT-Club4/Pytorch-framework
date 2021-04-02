import os

home = os.path.expanduser('~')

# 数据集类别数
NUM_CLASSES = 5

# 训练时batch的大小
BATCH_SIZE = 32

# 网络默认输入图像的大小
INPUT_SIZE = 300

# 训练最多额epoch
MAX_EPOCH = 100

# 使用GPU的数目
GPUS = 1

# 从第几个epoch开始训练，如果是0，从头开始
RESUME_EPOCH = 0

WEIGHT_DECAY = 5e-4

MOMENTUM = 0.9

# 初始学习率
LR = 1e-3

model_name = 'alexnet'

from models import Alexnet
from models import Resnet50, Resnet101, Resnet18, Resnet34, Resnet152
from models import Densenet121, Densenet169, Densenet161, Densenet201

MODEL_NAMES = {
    'alexnet': Alexnet,
    'resnet50': Resnet50,
    'resnet101': Resnet101,
    'resnet18': Resnet18,
    'resnet34': Resnet34,
    'resnet152': Resnet152,
    'densenet121': Densenet121,
    'densenet169': Densenet169,
    'densenet161': Densenet161,
    'densenet201': Densenet201
}

BASE = home + '/sht/pytorch-framework/data/'

# 训练好的模型保存位置
SAVE_FOLDER = BASE + 'weights/'

# 数据集存放位置
TRAIN_LABEL_DIR = BASE + 'train.txt'
VAL_LABEL_DIR = BASE + 'val.txt'
TEST_LABEL_DIR = BASE + 'test.txt'

# 训练完成，权重文件存放路径，默认保存在trained_model下
TRAINED_MODEL = BASE + 'weights/alexnet/epoch_100.pth'
