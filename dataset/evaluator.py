import os
import cv2
import sys
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class EvalDataset(Dataset):
    def __init__(self, img_dir_path, gt_dir_path, cuda):
        self.cuda = cuda
        self.trans = transforms.Compose([transforms.ToTensor()])
        if not os.path.exists(img_dir_path) or not os.path.exists(gt_dir_path):
            print('ERROR : dir path not exist!')
        # 图片的路径
        self.img_paths = [os.path.join(img_dir_path, file_name)
                           for file_name in sorted(os.listdir(img_dir_path))]
        # 图片标注的路径
        self.gt_paths = [os.path.join(gt_dir_path, file_name)
                         for file_name in sorted(os.listdir(gt_dir_path))]
        # 图片数量
        self.img_num = len(self.img_paths)


    def __getitem__(self, index: int):
        img = np.array(cv2.imread(self.img_paths[index]), dtype=np.float32)
        img -= np.array((104.00699, 116.66877, 122.67892))  # 减去BGR均值，该操作源自FCN。并不是归一化到[0,1]
        img = img.transpose((2, 0, 1))  # 转为(C,H,W)
        img = torch.from_numpy(img)


        gt = self.trans(Image.open(self.gt_paths[index]).convert('L'))
        if self.cuda:
            return img.cuda(), gt.cuda()
        return img, gt

    def __len__(self):
        return self.img_num


def calc_MAE(pred, gt):
    return torch.abs(pred - gt).mean()  # 一张图片的MAE


def calc_precision_recall(pred, gt, threshold=0.5):
    pred = pred.ge(threshold)
    gt = gt.ge(0.5)
    eps = sys.float_info.epsilon
    TP = torch.sum(torch.mul(pred, gt))
    P_in_pred = pred.sum()
    P_in_gt = gt.sum()
    p = (TP+eps)/(P_in_pred+eps)
    r = (TP+eps)/(P_in_gt+eps)
    return p.item(), r.item()


def calc_F_measure(p, r, beta2=0.3):
    return (1+beta2)*p*r / (beta2*p+r)
