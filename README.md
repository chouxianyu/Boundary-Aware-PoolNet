# Boundary-Aware-PoolNet

Boundary Aware PoolNet = PoolNet + BASNet : Deeply supervised PoolNet using the hybrid loss in BASNet for Salient Object Detection

Boundary Aware PoolNet = PoolNet + BASNet，即使用BASNet中的Deep Supervision和Hybrid Loss改进PoolNet。

经过评估，BAPoolNet的性能超过了之前的SOTA方法（注：在DUTS-TE数据集上进行测试，暂未在其它数据集上进行测试）。

# BAPoolNet结构

在PoolNet中Backbone是ResNet50时，模型自顶向下路径中有5个FUSE操作，我借鉴BASNet中的Deep Supervision和Hybrid Loss使用这5层输出的混合Loss之和进行梯度下降，我将这整个模型称为**BAPoolNet(Boundary Aware PoolNet)**，其结构如下图所示。

![img](https://pic4.zhimg.com/80/v2-7854dd376ab3ea8529d00b7056343067_720w.jpeg)

与PoolNet相比，BAPoolNet的不同之处为：

1. 添加5个边路输出以进行Deep Supervision
2. 在计算Loss时使用BCE损失、SSIM损失、IOU损失之和

除了对PoolNet的改进之外，BAPoolNet的其它实现细节和实验细节和PoolNet保持一致。

如何实现Boundary Aware PoolNet，具体请看代码。

# BAPoolNet代码

**传送门：**

- PoolNet代码：https://github.com/backseason/PoolNet
- BASNet代码：https://github.com/xuebinqin/BASNet

相比于PoolNet，BAPoolNet代码的改动之处有：

1. BCE Loss计算方法

	设置为`reduction=mean`而非`reduction=sum`，并且用`sigmoid+BCE`代替`F.binary_cross_entropy_with_logits`。

2. PoolNet`forward()`返回结果

	PoolNet类返回了5个边路输出而非最终输出

3. 整体Loss计算方法

	使用Hybrid Loss和Deep Supervision计算整体Loss

模型性能评估代码（MAE、F-measure等），我参考了：https://github.com/Hanqer/Evaluate-SOD

除了对PoolNet的改进之外，BAPoolNet的其它实现细节和实验细节和PoolNet保持一致。

**Coding Environments：**

```python
Python 3.7.3
torch 1.4.0
torchvision 0.5.0
tensorflow  2.0.0
tensorboard 2.0.2
tensorboardX 2.1
……
```

# BAPoolNet性能



## 5个边路输出可视化结果

![img](https://pic2.zhimg.com/80/v2-1ce1a1a0a3ad8f86bfd91c6755a6018a_720w.jpeg)

## 视觉对比

![img](https://pic4.zhimg.com/80/v2-b507a874646b5988e1e4a6b18c24ecf7_720w.jpeg)

## 量化对比

下表中MAE和maxF为各方法在DUTS-TE数据集上的测试结果。

|    Method     | Conference |  Backbone  | Size(MB) |   MAE↓    |   maxF↑   |
| :-----------: | :--------: | :--------: | :------: | :-------: | :-------: |
|    CapSal     |   CVPR19   | ResNet-101 |    -     |   0.063   |   0.826   |
|    PiCANet    |   CVPR18   | ResNet-50  |  197.2   |   0.050   |   0.860   |
|     DGRL      |   CVPR18   | ResNet-50  |  646.1   |   0.049   |   0.828   |
|    BASNet     |   CVPR19   | ResNet-34  |  348.5   |   0.047   |   0.860   |
|     U2Net     |   CVPR20   |    RSU     |  176.3   |   0.044   |   0.873   |
|      CPD      |   CVPR19   | ResNet-50  |  183.0   |   0.043   |   0.865   |
|    PoolNet    |   CVPR19   | ResNet-50  |  260.0   |   0.040   |   0.880   |
| **BAPoolNet** |     -      | ResNet-50  |  260.7   | **0.035** | **0.892** |

## PR曲线

下图为各方法在DUTS-TE数据集上的测试结果。

![img](https://pic4.zhimg.com/80/v2-e6ad03d7c136cf33416ad169bf0f89fc_720w.png)

## F-measure曲线

下图为各方法在DUTS-TE数据集上的测试结果。

![img](https://pic2.zhimg.com/80/v2-94a0c6ae4d9e04127fdc38ece5310ae7_720w.png)

如有问题，欢迎提issue。本人学识尚浅，期待共同进步！
