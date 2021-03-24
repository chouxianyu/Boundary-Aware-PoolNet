import torch.nn as nn
import torch
import torch.nn.functional as F
affine_par = True

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    """
    Shortcut Connections
    基本的Residual Block，并没有使用BottleNeck，即ResNet原文中ResNet-34的Residual Block
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, affine = affine_par)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, affine = affine_par)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out



class Bottleneck(nn.Module):
    """Shortcut Connections（Bottleneck）
    ResNet-50/101/152使用的Block
    The three layers are 1×1, 3×3, and 1×1 convolutions,
    where the 1×1 layers are responsible for reducing and then increasing (restoring) dimensions,
    leaving the 3×3 layer a bottleneck with smaller input/output dimensions
    """
    expansion = 4  # Bottleneck中通道数变化举例：输入256维，经1×1卷积得64，经3×3卷积得64，经1×1卷积得256。可知倍数是4

    def __init__(self, inplanes, planes, stride=1,  dilation_ = 1, downsample=None):
        """
        :params inplanes: 
        :params planes: 
        """
        super(Bottleneck, self).__init__()
        # 1×1卷积
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False) # change
        self.bn1 = nn.BatchNorm2d(planes, affine = affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        padding = 1
        if dilation_ == 2:
            padding = 2
        elif dilation_ == 4:
            padding = 4
        # 3×3卷积
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, # change
                               padding=padding, bias=False, dilation = dilation_)
        self.bn2 = nn.BatchNorm2d(planes,affine = affine_par)
        for i in self.bn2.parameters():
            i.requires_grad = False
        # 1×1卷积
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, affine = affine_par)
        for i in self.bn3.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        # 保存初始输入
        residual = x

        # 1×1卷积
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # 3×3卷积
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        # 1×1卷积
        out = self.conv3(out)
        out = self.bn3(out)

        # 对初始输入进行下采样
        if self.downsample is not None:
            residual = self.downsample(x)

        # Shortcut Connections
        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    """生成torch版本的ResNet
    ResNet50 Backbone分2个部分：①7×7卷积及其BN、ReLU、Pooling；②4个阶段的卷积层
    """
    def __init__(self, block, layers):
        """
        :params block: 是1个类（ResNet50: Bottleneck, ResNet34: BasicBlock）
        :params layers: 是1个列表[3, 4, 6, 3]，即ResNet各阶段卷积层中Block的数量
        """
        self.inplanes = 64  
        super(ResNet, self).__init__()
        
        # ResNet50 Backbone第1部分
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7,stride=2, padding=3, bias=False)  # 7×7卷积
        self.bn1 = nn.BatchNorm2d(64,affine = affine_par) # BN层
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True) # ReLU
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True) # change
     
        # ResNet50 Backbone第2部分（4个阶段的卷积层）
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation__ = 2)

        # 卷积层和BN层初始化
        for m in self.modules():  # 变量网络中的所有module
            if isinstance(m, nn.Conv2d):  # 初始化卷积层参数
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)  # 用正态分布初始化
            elif isinstance(m, nn.BatchNorm2d):  # 初始化BatchNorm层参数
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation__ = 1):
        """
        :params block: 是1个类（ResNet50: Bottleneck, ResNet34: BasicBlock）
        :params planes: 
        :params blocks: block的数量
        :params layers: 是1个列表[3, 4, 6, 3]，即ResNet各阶段卷积层中Block的数量
        :params stride: 不只是downsample用到了（short connection时若输入输出尺寸不同，stride=2使输入尺寸减半，与输出尺寸匹配），layers也用到了（234阶段第1个block中第1个卷积层需要使尺寸减半）
        """
        downsample = None # 在ResNet Backbone第2部分中，第2、3、4阶段卷积中的第1个Block都需要进行1×1卷积以调整通道数量差异（其中通过步长=2使特征图尺寸减半）
        if stride != 1 or self.inplanes != planes * block.expansion or dilation__ == 2 or dilation__ == 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion,affine = affine_par),
            )
        for i in downsample._modules['1'].parameters():
            i.requires_grad = False

        # Block数组
        layers = []
        layers.append(block(self.inplanes, planes, stride,dilation_=dilation__, downsample = downsample ))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation_=dilation__))

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        返回5个输出
        """
        tmp_x = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        tmp_x.append(x)  # ResNet Backbone第1部分（7×7卷积）后的输出
        x = self.maxpool(x)

        x = self.layer1(x)
        tmp_x.append(x)  # ResNet Backbone第2部分第1阶段卷积的输出
        x = self.layer2(x)
        tmp_x.append(x)  # ResNet Backbone第2部分第2阶段卷积的输出
        x = self.layer3(x)
        tmp_x.append(x)  # ResNet Backbone第2部分第3阶段卷积的输出
        x = self.layer4(x)
        tmp_x.append(x)  # ResNet Backbone第2部分第4阶段卷积的输出

        return tmp_x


class ResNet_locate(nn.Module):
    """包含resnet的backbone、GGM(PPM和GGF)，forward后返回resnet的各层输出和GGM的各层输出。
    ->resnet
    ->PPM4个分支(各个分支大概操作为AdaptiveAvgPool、Conv、ReLU)->通过上采样改变4分支输出的尺寸到resnet顶层输出的尺寸->4分支的通道拼接->通过卷积改变通道数为resnet顶层输出的通道数、ReLU
    ->通过上采样改变尺寸到resnet对应层的尺寸->通过卷积改变通道数为resnet对应层输出的通道数、ReLU
    """
    def __init__(self, block, layers):
        """
        :params block: 是1个类（ResNet50: Bottleneck, ResNet34: BasicBlock）
        :params layers: 是1个列表[3, 4, 6, 3]，即ResNet各阶段卷积层中Block的数量
        """
        super(ResNet_locate,self).__init__()
        self.resnet = ResNet(block, layers)  # 定义ResNet50网络
        self.in_planes = 512  # PPM的输入通道数
        self.out_planes = [512, 256, 256, 128]  # 4个GGF的输出通道数，从右到左

        # FPN在从bottle-up路径向top-down路径转换时需要进行1×1卷积（改变通道数）
        self.ppms_pre = nn.Conv2d(2048, self.in_planes, 1, 1, bias=False)

        ppms, infos = [], []
        for ii in [1, 3, 5]: # ii为1时即为全局平均池化
            # nn.AdaptiveAvgPool2d(ii)： 输出的尺寸为(ii,ii)
            # nn.Conv2d(self.in_planes, self.in_planes, 1, 1, bias=False)：输入输出通道数都为self.in_planes、kernal_size=1、stride=1
            ppms.append(nn.Sequential(nn.AdaptiveAvgPool2d(ii), nn.Conv2d(self.in_planes, self.in_planes, 1, 1, bias=False), nn.ReLU(inplace=True)))
        # PPM中的3个分支：1个global average pooling layer、2个adaptive average pooling layer
        self.ppms = nn.ModuleList(ppms)

        # 将PPM的4个分支提取到的特征进行融合，*4是因为PPM共有4个分支
        self.ppm_cat = nn.Sequential(nn.Conv2d(self.in_planes * 4, self.in_planes, 3, 1, 1, bias=False), nn.ReLU(inplace=True))
        for ii in self.out_planes:
            infos.append(nn.Sequential(nn.Conv2d(self.in_planes, ii, 3, 1, 1, bias=False), nn.ReLU(inplace=True)))
        self.infos = nn.ModuleList(infos)  # GGF

        for m in self.modules():  # 卷积层BatchNorm2d和初始化
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def load_pretrained_model(self, model):
        """加载预训练模型"""
        self.resnet.load_state_dict(model, strict=False)

    def forward(self, x):
        """返回
        xs：resnet的从左到右5个金字塔层的输出
        infos：GGM的从右到左4层的输出
        """
        x_size = x.size()[2:]
        xs = self.resnet(x)  # resnet的5层输出（具体见ResNet类中的forward函数）

        # 取出resnet最后1层输出进行PPM的预处理，会传入PPM的1个分支
        xs_1 = self.ppms_pre(xs[-1])

        # PPM4个分支的输出
        xls = [xs_1]  # identity mapping layer
        for k in range(len(self.ppms)):
            xls.append(F.interpolate(self.ppms[k](xs_1), xs_1.size()[2:], mode='bilinear', align_corners=True))
        
        # PPM的输出（尺寸和Backone顶层输出的尺寸一致）
        xls = self.ppm_cat(torch.cat(xls, dim=1)) # 将PPM的4个分支的输出在通道维度上拼接，然后通过卷积改变通道数为resnet最后1层输出的通道数、ReLU

        infos = []
        for k in range(len(self.infos)):  # GGF的输出
            infos.append(self.infos[k](F.interpolate(xls, xs[len(self.infos) - 1 - k].size()[2:], mode='bilinear', align_corners=True)))

        return xls, xs, infos

def resnet50_locate():
    model = ResNet_locate(Bottleneck, [3, 4, 6, 3])
    return model
