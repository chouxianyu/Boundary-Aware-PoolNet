import torch
import torch.nn as nn
from hybrid_loss import pytorch_ssim
from hybrid_loss import pytorch_iou


bce_loss = nn.BCELoss(reduction='mean')  # bce_loss = nn.BCELoss(size_average=True)
ssim_loss = pytorch_ssim.SSIM(window_size=11, size_average=True)
iou_loss = pytorch_iou.IOU(size_average=True)


def bce_ssim_iou_loss(pred, target):
    """返回1个1维的tensor数组：BCE Loss、SSIM Loss、IOU Loss、3者之和"""
    bce_out = bce_loss(pred, target)  # 结果是0维tensor，即标量
    ssim_out = 1 - ssim_loss(pred, target)  # 结果是0维tensor，即标量
    iou_out = iou_loss(pred, target) # 结果是0维tensor，即标量

    hybrid_loss = bce_out + ssim_out + iou_out

    return torch.stack([bce_out, ssim_out, iou_out, hybrid_loss]) # 结果是1个1维tensor


def muti_loss_fusion(side_outputs, labels, w=[1,1,1,1,1]):
    """
    默认Backbone为ResNet50，所以side_outputs中有5个边路输出
    计算5层的4种loss及5层的loss之和，可以看做6层的4个loss
    最终返回1个6×4的tensor，6指top-down路径从上到下的5层及其之和，4指3种独立loss及其之和
    """
    losses0 = bce_ssim_iou_loss(side_outputs[0], labels)
    losses1 = bce_ssim_iou_loss(side_outputs[1], labels)
    losses2 = bce_ssim_iou_loss(side_outputs[2], labels)
    losses3 = bce_ssim_iou_loss(side_outputs[3], labels)
    losses4 = bce_ssim_iou_loss(side_outputs[4], labels)
    losses5 = w[0]*losses0 + w[1]*losses1 + w[2]*losses2 + w[3]*losses3 + w[4]*losses4
 
    return torch.stack([losses0, losses1, losses2, losses3, losses4, losses5])
