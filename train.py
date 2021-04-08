import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from data import train_dataloader, train_datasets
import cfg
from utils import adjust_learning_rate_cosine, adjust_learning_rate_step

# 创建训练模型参数保存文件及
save_folder = cfg.SAVE_FOLDER + cfg.model_name
os.makedirs(save_folder, exist_ok=True)


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']  # 提取网络结构
    model.load_state_dict(checkpoint['model_state_dict'])  # 加载网络权重参数
    return model


if not cfg.RESUME_EPOCH:
    print('****** Training {} ****** '.format(cfg.model_name))
    print('****** loading the pretrained weights ****** ')
    if not cfg.model_name.startswith('efficientnet'):
        model = cfg.MODEL_NAMES[cfg.model_name](num_classes=cfg.NUM_CLASSES)
        for child in model.children():
            print(child)
            for param in child.parameters():
                param.requires_grad = True
    # efficientnet特有，build_model有所不同
    else:
        model = cfg.MODEL_NAMES[cfg.model_name](cfg.model_name, num_classes=cfg.NUM_CLASSES)
        for child in model.children():
            print(child)
            for param in child.parameters():
                param.requires_grad = True
else:
    print(' ******* Resume training from {}  epoch {} *********'.format(cfg.model_name, cfg.RESUME_EPOCH))
    model = load_checkpoint(os.path.join(save_folder, 'epoch_{}.pth'.format(cfg.RESUME_EPOCH)))

# 进行多GPU并行计算
if cfg.GPUS > 1:
    print('****** using multiple gpus to training ********')
    model = nn.DataParallel(model, device_ids=list(range(cfg.GPUS)))
else:
    print('****** using single gpu to training ********')
print("...... Initialize the network done!!! .......")

# 模型放置在GPU上进行计算
if torch.cuda.is_available():
    model.cuda()

# 定义优化器和损失函数
# optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.LR)
optimizer = optim.Adam(model.parameters(), lr=cfg.LR)
# optimizer = optim.SGD(model.parameters(), lr=cfg.LR, momentum=cfg.MOMENTUM, weight_decay=cfg.WEIGHT_DECAY)
criterion = nn.CrossEntropyLoss()
# criterion = nn.BCEWithLogitsLoss()
# criterion = LabelSmoothSoftmaxCE()
# criterion = LabelSmoothingCrossEntropy()

lr = cfg.LR

batch_size = cfg.BATCH_SIZE

# 每个epoch含有多少个batch
epoch_size = len(train_datasets) // batch_size
# 训练cfg.MAX_EPOCH个epoch
max_iter = cfg.MAX_EPOCH * epoch_size
start_iter = cfg.RESUME_EPOCH * epoch_size
epoch = cfg.RESUME_EPOCH

# cosine学习率调整参数
warmup_epoch = 5
warmup_step = warmup_epoch * epoch_size
global_step = 0
# step学习率调整参数
stepvalues = (10 * epoch_size, 20 * epoch_size, 30 * epoch_size)
step_index = 0

model.train()
for iteration in range(start_iter, max_iter):
    global_step += 1
    # 更新迭代器
    if iteration % epoch_size == 0:
        batch_iterator = iter(train_dataloader)
        loss = 0
        epoch += 1
        # 保存模型
        if epoch % 100 == 0 and epoch > 0:  # 暂时调整为100，节省存储空间，方便测试模型
            if cfg.GPUS > 1:
                checkpoint = {
                    'model': model.module,
                    'model_state_dict': model.module.state_dict(),
                    'epoch': epoch
                }
                torch.save(checkpoint, os.path.join(save_folder, 'epoch_{}.pth'.format(epoch)))
            else:
                checkpoint = {
                    'model': model,
                    'model_state_dict': model.state_dict(),
                    'epoch': epoch
                }
                torch.save(checkpoint, os.path.join(save_folder, 'epoch_{}.pth'.format(epoch)))
    if iteration in stepvalues:
        step_index += 1

    # 选择何种学习率调整方式
    lr = adjust_learning_rate_step(optimizer, cfg.LR, 0.1, epoch, step_index, iteration, epoch_size)
    # lr = adjust_learning_rate_cosine(optimizer, global_step=global_step, learning_rate_base=cfg.LR, total_steps=max_iter, warmup_steps=warmup_steps)

    images, labels = next(batch_iterator)
    if torch.cuda.is_available():
        images, labels = images.cuda(), labels.cuda()

    out = model(images)
    loss = criterion(out, labels)

    optimizer.zero_grad()  # 清空梯度信息，否则在每次进行反向传播时都会累加
    loss.backward()
    optimizer.step()

    prediction = torch.max(out, 1)[1]
    train_correct = (prediction == labels).sum()
    train_acc = (train_correct.float()) / batch_size

    if iteration % 10 == 0:
        print('Epoch:' + repr(epoch)
              + ' || epochiter: '
              + repr(iteration % epoch_size)
              + '/' + repr(epoch_size)
              + '|| Totel iter '
              + repr(iteration)
              + '|| Loss: %.6f||' % (loss.item())
              + 'ACC: %.3f ||' % (train_acc * 100)
              + 'LR: %.8f' % lr)
