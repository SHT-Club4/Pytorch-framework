import torch
import os
from PIL import Image
import pandas as pd
from tqdm import tqdm
import numpy as np
from collections import Counter
import cfg
from data import get_test_transform


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    # print(checkpoint)
    model = checkpoint['model']  # 提取网络结构
    print(model)
    model.load_state_dict(checkpoint['model_state_dict'])  # 加载网络权重参数
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()
    return model


def predict(model):
    # 读入模型
    model = load_checkpoint(model)
    print('..... Finished loading model! ......')
    if torch.cuda.is_available():
        model.cuda()
    pred_list, _id = [], []
    for i in tqdm(range(len(imgs))):
        img_path = imgs[i].strip()
        _id.append(os.path.basename(img_path).split('.')[0])
        img = Image.open(img_path).convert('RGB')
        img = get_test_transform(size=cfg.INPUT_SIZE)(img).unsqueeze(0)
        if torch.cuda.is_available():
            img = img.cuda()
        with torch.no_grad():
            out = model(img)
        prediction = torch.argmax(out, dim=1).cpu().item()
        pred_list.append(prediction)
    return _id, pred_list


if __name__ == '__main__':
    trained_model = cfg.TRAINED_MODEL
    model_name = cfg.model_name
    with open(cfg.TEST_LABEL_DIR, 'r') as f:
        imgs = f.readlines()
    _id, pred_list = predict(trained_model)
    submission = pd.DataFrame({"ID": _id, "Label": pred_list})
    submission.to_csv(cfg.RESULTS + '{}_submission.csv'.format(model_name), index=False, header=False)
