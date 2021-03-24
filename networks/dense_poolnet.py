import torch
from torch import nn
import torch.nn.functional as F
from .deeplab_resnet import resnet50_locate


# resnet50的设置
config_resnet = {'convert': [[64,256,512,1024,2048],[128,256,256,512,512]], 'deep_pool': [[512, 512, 256, 256, 128], [512, 256, 256, 128, 128], [False, True, True, True, False], [True, True, True, True, False]], 'score': [512, 256, 256, 128, 128]}


class ConvertLayer(nn.Module):
    """
    resnet会用到convert层
    ConvertLayer用来改变resnet的5层输出的通道数
    """
    def __init__(self, list_k):
        """
        list_k即config_resnet['convert']，1个2行5列的列表，第1行是in_channels，第2行是out_channels
        """
        super(ConvertLayer, self).__init__()
        up = []
        for i in range(len(list_k[0])):
            up.append(nn.Sequential(nn.Conv2d(list_k[0][i], list_k[1][i], 1, 1, bias=False), nn.ReLU(inplace=True))) # 1×1卷积
        self.convert0 = nn.ModuleList(up) # 5个1×1卷积层，每个卷积层后面有ReLU

    def forward(self, list_x):
        """list_x是backbone的金字塔层输出，ConvertLayer根据配置改变其通道数
        """
        resl = []
        for i in range(len(list_x)):
            resl.append(self.convert0[i](list_x[i]))
        return resl


class DeepPoolLayer(nn.Module):
    """FAM & FPN FUSE操作"""
    def __init__(self, k, k_out, need_x2, need_fuse):
        """
        k: FAM输入的通道数
        k_out: FAM输出的通道数
        need_x2: 是否有backbone第i-1个金字塔层(从左到右编号)
        need_fuse: 是否有GGF的输出
        """
        super(DeepPoolLayer, self).__init__()
        self.pools_sizes = [2,4,8] # 下采样比例
        self.need_x2 = need_x2
        self.need_fuse = need_fuse
        pools, convs = [],[]
        for i in self.pools_sizes:
            pools.append(nn.AvgPool2d(kernel_size=i, stride=i))
            convs.append(nn.Conv2d(k, k, 3, 1, 1, bias=False))
        self.pools = nn.ModuleList(pools) # 平均下采样/平均池化
        self.convs = nn.ModuleList(convs) # 3×3卷积层，有padding所以特征尺寸不会改变
        self.relu = nn.ReLU()
        self.conv_sum = nn.Conv2d(k, k_out, 3, 1, 1, bias=False) # FAM中sum后的3×3卷积，有padding
        if self.need_fuse:
            self.conv_sum_c = nn.Conv2d(k_out, k_out, 3, 1, 1, bias=False)  # FPN操作中的3×3卷积，有padding

    def forward(self, x, x2=None, x3=None):
        """
        merge = self.deep_pool[0](conv2merge[0], conv2merge[1], infos[0])
        x: backbone第i个金字塔层的输出（假设从左到右/从浅到深编号）
        x2: backbone第i-1个金字塔层的输出
        x3 : 对应GGF的输出
        """
        x_size = x.size()
        # FAM
        resl = x
        for i in range(len(self.pools_sizes)):
            y = self.convs[i](self.pools[i](x)) # 将输入先后进行平均池化、3×3卷积
            resl = torch.add(resl, F.interpolate(y, x_size[2:], mode='bilinear', align_corners=True)) # 上采样后4个分支相加
        resl = self.relu(resl) # ReLU
        # 如果有x2，即需要FPN FUSE操作，就需要上采样，以使backbone第i个金字塔层和第i-1个金字塔层尺寸相同(从左到右编号)
        if self.need_x2:
            resl = F.interpolate(resl, x2.size()[2:], mode='bilinear', align_corners=True)
        resl = self.conv_sum(resl) # FAM中sum操作后的3×3卷积
        if self.need_fuse:
            resl = self.conv_sum_c(torch.add(torch.add(resl, x2), x3)) # FPN操作：3者求和然后3×3卷积(有padding)
        return resl


class ScoreLayer(nn.Module):
    """生成预测图：先通过1×1卷积将通道数改为1，然后通过上采样将尺寸改为输入的尺寸"""
    def __init__(self, k):
        """
        k: 输入通道数(即从top-down路径中从右到左deep_pool操作输出的通道数)，Backbone为resnet50时，k先后为[512,512,256,256,128]
        """
        super(ScoreLayer, self).__init__()
        self.score = nn.Conv2d(k ,1, 1, 1) # 输出1个通道，1×1卷积

    def forward(self, x, x_size=None):
        x = self.score(x) # 卷积到1个通道
        if x_size is not None:
            x = F.interpolate(x, x_size[2:], mode='bilinear', align_corners=True) # 上采样到模型输入的尺寸
        return torch.sigmoid(x)


def extra_layer(base_model_cfg, base):
    """创建定义PoolNet需要的所有Layer并返回"""
    # 确定用哪个backbone
    if base_model_cfg == 'resnet':
        config = config_resnet
    
    convert_layers, deep_pool_layers, score_layers = [], [], []
    # 创建ConvertLayer
    convert_layers = ConvertLayer(config['convert'])  # 在ConvertLayer类中会使用nn.ModuleList
    # 创建DeepPoolLayer, 在PoolNet类中会使用nn.ModuleList
    for i in range(len(config['deep_pool'][0])):
        deep_pool_layers += [DeepPoolLayer(config['deep_pool'][0][i], config['deep_pool'][1][i], config['deep_pool'][2][i], config['deep_pool'][3][i])]
    # 创建ScoreLayer
    score_layers = [ScoreLayer(k) for k in config['score']]

    return base, convert_layers, deep_pool_layers, score_layers


class PoolNet(nn.Module):
    def __init__(self, base_model_cfg, base, convert_layers, deep_pool_layers, score_layers):
        super(PoolNet, self).__init__()
        self.base_model_cfg = base_model_cfg  # 一个字符串，'resnet'
        self.base = base # vgg16或者resnet50本身及其GGM
        self.deep_pool = nn.ModuleList(deep_pool_layers) # FAM模块
        self.score_layers = nn.ModuleList(score_layers)
        if self.base_model_cfg == 'resnet':
            self.convert = convert_layers # 改变resnet5层金字塔层输出的通道数：[64,256,512,1024,2048] ==>  [128,256,256,512,512]

    def forward(self, x):
        x_size = x.size() # 输入图片的尺寸
        _, conv2merge, infos = self.base(x) # 获取PPM(512通道)、backbone5个金字塔层的输出、GGM的4路输出
        if self.base_model_cfg == 'resnet':
            # 改变resnet5层金字塔层输出的通道数：[64,256,512,1024,2048] ==>  [128,256,256,512,512]
            conv2merge = self.convert(conv2merge)
        conv2merge = conv2merge[::-1] # resnet5层输出倒序：从左到右 => 从右到左

        # 从右到左保存deep_pool(FAM+FPN)操作的输出，后续计算score和loss，实现densely supervise
        side_outputs = []  # 通道数为[512, 256, 256, 128, 128]
        # 进行FAM和FPN操作
        merge = self.deep_pool[0](conv2merge[0], conv2merge[1], infos[0]) # 最右侧的1个FAM+FPN（有GGF输入和前1个金字塔层）
        side_outputs.append(merge)
        for k in range(1, len(conv2merge)-1):  # 剩余2或3个FAM+FPN（有GGF输入和前1个金字塔层）
            merge = self.deep_pool[k](merge, conv2merge[k+1], infos[k])
            side_outputs.append(merge)
        merge = self.deep_pool[-1](merge)  # 最后1个FAM+FPN（无GGF输入和前1个金字塔层）
        side_outputs.append(merge)
        # 分数层
        scores = [self.score_layers[i](side_outputs[i], x_size) for i in range(len(side_outputs))]

        return scores  # 得到5个与原图尺寸一样的score map，最后1个就是最终预测的显著性图
        
         
def build_model(base_model_cfg='resnet'):
    if base_model_cfg == 'resnet':
        return PoolNet(base_model_cfg, *extra_layer(base_model_cfg, resnet50_locate()))


def weights_init(m):
    """将m（一个torch.nn.Module）的weight和bias初始化
    """
    if isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0, 0.01) # 用均值为0,、标准差为0.01的正态分布将该Module的weight初始化
        if m.bias is not None: # 如果该Module有bias就都赋值为0
            m.bias.data.zero_()
