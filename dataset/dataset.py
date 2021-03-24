import os
from PIL import Image
import cv2
import torch
from torch.utils import data
from torchvision.transforms import functional as F
import numpy as np
import random


class ImageDataTrain(data.Dataset):
    """表示训练集和验证集（有原图片和label）
    """
    def __init__(self, data_root, data_list):
        self.sal_root = data_root  # 训练集图片根目录路径
        self.sal_source = data_list  # 训练集图片列表文件的路径

        with open(self.sal_source, 'r') as f:
            self.sal_list = [x.strip() for x in f.readlines()]  # 训练集图片名称列表

        self.sal_num = len(self.sal_list)  # 训练集图片数量


    def __getitem__(self, item):
        """该函数读取训练集中的一张图片及其label
        :param item: int, 图片的index
        """
        im_name = self.sal_list[item % self.sal_num].split()[0]  # 图片名称
        gt_name = self.sal_list[item % self.sal_num].split()[1]  # GT名称
        sal_image = load_image(os.path.join(self.sal_root, im_name))  # 加载图片
        sal_label = load_sal_label(os.path.join(self.sal_root, gt_name))  # 加载label
        sal_image, sal_label = cv_random_flip(sal_image, sal_label)  # 随机水平翻转
        sal_image = torch.Tensor(sal_image)  # 转为tensor
        sal_label = torch.Tensor(sal_label)  # 转为tensor

        sample = {'sal_image': sal_image, 'sal_label': sal_label} # 将图片和label合并到字典中，图片和label都是CHW
        return sample

    def __len__(self):
        return self.sal_num


class ImageDataTest(data.Dataset):
    """
        表示测试集（只有图片没有label）
    """
    def __init__(self, data_root, data_list):
        self.data_root = data_root  # 测试集图片根目录路径
        self.data_list = data_list  # 测试集图片列表文件的路径
        with open(self.data_list, 'r') as f:
            self.image_list = [x.strip() for x in f.readlines()]  # 测试集图片名称列表

        self.image_num = len(self.image_list)  # 测试集图片数量

    def __getitem__(self, item):
        image, im_size = load_image_test(os.path.join(self.data_root, self.image_list[item])) # 加载测试集图片
        image = torch.Tensor(image) # 转为tensor

        return {'image': image, 'name': self.image_list[item % self.image_num], 'size': im_size} # 将图片本身、图片名称和图片尺寸合并到字典中

    def __len__(self):
        return self.image_num


def get_loader(config, mode='train', pin=False):
    """
        生成训练用或测试用的dataloader
    """
    shuffle = False # 是否打乱顺序
    if mode == 'train':
        shuffle = True
        dataset = ImageDataTrain(config.train_root, config.train_list)
        data_loader = data.DataLoader(dataset=dataset, batch_size=config.batch_size, shuffle=shuffle, num_workers=config.num_thread, pin_memory=pin)
    else:
        dataset = ImageDataTest(config.test_root, config.test_list)
        data_loader = data.DataLoader(dataset=dataset, batch_size=config.batch_size, shuffle=shuffle, num_workers=config.num_thread, pin_memory=pin)
    return data_loader


def load_image(path):
    """
        用cv2加载(H,W,C)格式的图片，然后减去BGR均值，然后转为(C,H,W)格式的图片并返回
    """
    if not os.path.exists(path):
        print('File {} not exists'.format(path))
    im = cv2.imread(path)  # (H,W,C)的图片
    in_ = np.array(im, dtype=np.float32)
    in_ -= np.array((104.00699, 116.66877, 122.67892))  # 减去BGR均值，该操作源自FCN。并不是归一化到[0,1]
    in_ = in_.transpose((2,0,1)) # 转为(C,H,W)
    return in_


def load_image_test(path):
    """
        用cv2加载(H,W,C)格式的图片，然后减去BGR均值，然后转为(C,H,W)格式的图片
        返回(C,H,W)格式的图片和图片尺寸（元组）
    """
    if not os.path.exists(path):
        print('File {} not exists'.format(path))
    im = cv2.imread(path)  # (H,W,C)的图片
    in_ = np.array(im, dtype=np.float32)
    im_size = tuple(in_.shape[:2])
    in_ -= np.array((104.00699, 116.66877, 122.67892))  # 减去BGR均值，该操作源自FCN
    in_ = in_.transpose((2, 0, 1))  # 转为(C,H,W)
    return in_, im_size


def load_sal_label(path):
    """
        用PIL加载JPG:(H,W,C)格式或PNG:(H,W)的label图片，然后(H,W,C) => (H,W)，然后[0,255] => [0,1]，然后(H,W) => (1,H,W)
    """
    if not os.path.exists(path):
        print('File {} not exists'.format(path))
    im = Image.open(path)  # JPG图片会得到类别为`PIL.JpegImagePlugin.JpegImageFile`的对象，PNG图片则得到类别为`PIL.PngImagePlugin.PngImageFile`的对象
    label = np.array(im, dtype=np.float32) # 转为numpy的array
    if len(label.shape) == 3:  # 消除JPG图片的通道
        label = label[:,:,0]
    label = label / 255.  # [0,255] => [0,1]
    label = label[np.newaxis, ...]  # (H,W) => (1,H,W)
    return label


def cv_random_flip(img, label):
    """
        随机水平翻转图片和label
    """
    flip_flag = random.randint(0, 1)
    if flip_flag == 1:
        img = img[:,:,::-1].copy()
        label = label[:,:,::-1].copy()
    return img, label
