import os
import glob
import sys
import cfg
import random
sys.path.append("..")

if __name__ == '__main__':
    train_data_path = cfg.BASE + 'train/'
    labels = os.listdir(train_data_path)
    test_data_path = cfg.BASE + 'test/'
    # 写train.txt文件
    txt_path = cfg.BASE
    # print(labels)
    for index, label in enumerate(labels):
        img_list = glob.glob(os.path.join(train_data_path, label, '*.jpg'))
        # print(img_list)
        random.shuffle(img_list)
        print(len(img_list))
        train_list = img_list[:int(0.8 * len(img_list))]
        val_list = img_list[(int(0.8 * len(img_list)) + 1):]
        with open(txt_path + 'train.txt', 'a') as f:
            for img in train_list:
                f.write(img + ' ' + str(index))
                f.write('\n')
        with open(txt_path + 'val.txt', 'a') as f:
            for img in val_list:
                f.write(img + ' ' + str(index))
                f.write('\n')
    img_list = glob.glob(os.path.join(test_data_path, '*.jpg'))
    with open(txt_path + 'test.txt', 'a') as f:
        for img in img_list:
            f.write(img)
            f.write('\n')
