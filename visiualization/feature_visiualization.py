import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import sys
sys.path.append("..")
import cfg
from data import get_test_transform


# 对于给定的一个网络层输出x，x为numpy格式的array，维度为[0, channels, width, height]
def draw_features(width, height, channels, x, save_name):
    fig = plt.figure(figsize=(32, 32))
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.05, hspace=0.05)
    for i in range(channels):
        plt.subplot(height, width, i + 1)
        plt.axis('off')
        img = x[0, i, :, :]
        p_min = np.min(img)
        p_max = np.max(img)
        img = (img - p_min) / (p_max - p_min + 1e-6)
        plt.imshow(img, cmap='gray')
    fig.savefig(save_name, dpi=300)
    fig.clf()
    plt.close()


# 读取模型
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['model_state_dict'])
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    return model


# 输出特征图存放路径
save_path = './'


def predict(model):
    model = load_checkpoint(model)
    print('.....Finishing loading model!.....')
    if torch.cuda.is_available():
        model.cuda()
    img = Image.open(img_path).convert('RGB')
    img = get_test_transform(size=cfg.INPUT_SIZE)(img).unsqueeze(0)
    if torch.cuda.is_available():
        img = img.cuda()
    with torch.no_grad():
        # 以vgg11为例
        # 不同网络结构要做相应的修改
        x = model.features(img)
        draw_features(8, 8, 64, x.cpu().numpy(), "{}/features.jpg".format(save_path))


if __name__ == '__main__':
    trained_model = cfg.TRAINED_MODEL  # 用于可视化的权重文件路径
    img_path = './test.jpg'  # 测试图片路径
    predict(trained_model)
