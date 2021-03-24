import torch
from torch.nn import functional as F
from torch.optim import Adam
from torch.autograd import Variable
from networks.dense_poolnet import build_model, weights_init
import numpy as np
import os
import cv2
import time
from hybrid_loss import muti_loss_fusion
from tensorboardX import SummaryWriter
from dataset import evaluator


class Solver(object):
    def __init__(self, train_loader, test_loader, eval_loader: evaluator.EvalDataset, config):
        self.train_loader = train_loader  # 训练集(验证集)loader
        self.test_loader = test_loader  # 测试集loader
        self.eval_loader = eval_loader # 验证loader
        self.config = config
        self.best_mae = 100
        self.best_F_measure = -1
        self.best_epoch = 0
        if train_loader is not None:
            self.logger = SummaryWriter(logdir=config.save_folder+'/tblog')
            self.lossType = ['BCE', 'SSIM', 'IOU', 'FUSE']
        self.iter_size = config.iter_size 
        self.lr_decay_epoch = [8, 16] # 学习率衰减epoch
        self.build_model() # 构建网络
        if config.mode == 'test':
            print('Loading pre-trained model from %s...' % self.config.model)
            if self.config.cuda:
                self.net.load_state_dict(torch.load(self.config.model))
            else:
                self.net.load_state_dict(torch.load(self.config.model, map_location='cpu'))
            self.net.eval()

    # print the network information and parameter numbers
    def print_network(self, model, name):
        num_params = 0
        for p in model.parameters():  # 计算所有参数中元素的个数
            num_params += p.numel()
        print(name)
        print(model)
        print("The number of parameters: {}".format(num_params))

    # build the network
    def build_model(self):
        self.net = build_model(self.config.arch) # 构建PoolNet网络结构
        if self.config.cuda:
            self.net = self.net.cuda()
        self.net.eval()  # use_global_stats = True
        self.net.apply(weights_init) # 模型参数初始化
        if self.config.load == '': # 加载backbone预训练模型
            self.net.base.load_pretrained_model(torch.load(self.config.pretrained_model))
        else:  # 加载poolnet预训练模型
            self.net.load_state_dict(torch.load(self.config.load))

        self.lr = self.config.lr # 学习率
        self.wd = self.config.wd # weight decay

        self.optimizer = Adam(filter(lambda p: p.requires_grad, self.net.parameters()), lr=self.lr, weight_decay=self.wd) # 构建优化器
        self.print_network(self.net, 'PoolNet Structure') # 输出网络结构

    # 测试，仅保存indexes指定通道的输出
    def test(self, indexes):
        mode_name = 'test'
        time_s = time.time()
        img_num = len(self.test_loader)
        for i, data_batch in enumerate(self.test_loader):
            images, name, im_size = data_batch['image'], data_batch['name'][0], np.asarray(data_batch['size'])
            with torch.no_grad():
                images = Variable(images)
                if self.config.cuda:
                    images = images.cuda()
                preds = self.net(images)
                for j in indexes:
                    pred = np.squeeze(preds[j].cpu().data.numpy())
                    multi_fuse = 255 * pred
                    cv2.imwrite(os.path.join(self.config.test_fold, name[:-4] + '_' + mode_name + '_' + str(j) + '.png'), multi_fuse)
        time_e = time.time()
        print('Speed: %f FPS' % (img_num/(time_e-time_s)))
        print('Test Done!')
 
    # training phase(hybrid loss & deep supervision)
    def train(self):
        batch_num = len(self.train_loader.dataset) // self.config.batch_size# batch的数量
        aveGrad = 0  # 在当前mini-batch中，训练过的图片的数量(前提为batch_size为1)
        iter_cnt = 0 # 训练过程中当前进行了多少个iteration
        for epoch in range(self.config.epoch):
            batch_avg_loss4 = 0  # 该mini-batch中每张图片的平均loss(最后1个side output的loss)
            batch_avg_loss5 = 0  # 该mini-batch中每张图片的平均loss(所有side output的loss之和)
            self.net.zero_grad()
            for i, data_batch in enumerate(self.train_loader):
                sal_image, sal_label = data_batch['sal_image'], data_batch['sal_label'] # 取出img和label
                if (sal_image.size(2) != sal_label.size(2)) or (sal_image.size(3) != sal_label.size(3)): # 检查img和label的宽和高是否相等
                    print('IMAGE ERROR, PASSING```')
                    continue
                sal_image, sal_label= Variable(sal_image), Variable(sal_label) # 变换成Variable
                if self.config.cuda: # 检查CUDA
                    # cudnn.benchmark = True
                    sal_image, sal_label = sal_image.cuda(), sal_label.cuda()

                sal_preds = self.net(sal_image)
                losses = muti_loss_fusion(sal_preds, sal_label)  # 该batch图片的6×4种loss
                batch_avg_loss4 += losses[4][3].item() / (self.iter_size * self.config.batch_size)  # 该mini-batch中每张图片的平均loss(最后1个side output的loss)
                batch_avg_loss5 += losses[5][3].item() / (self.iter_size * self.config.batch_size)  # 该mini-batch中每张图片的平均loss(所有side output的loss之和)
                losses[5][3].backward()  # 计算该batch的梯度（用多层的混合loss计算梯度）

                aveGrad += 1 # iteration次数加1
                iter_cnt += 1  # iteration次数加1

                # accumulate gradients as done in DSS，每个batch更新1次参数并将梯度清空，同时输出日志
                if aveGrad % self.iter_size == 0: # 因为batch_size只能是1，则iter_size就相当于真正的batch_size
                    # 使用TensorboardX记录训练日志
                    self.tb_log(data=losses.detach() / (self.iter_size * self.config.batch_size), step=iter_cnt, mode='layer&loss')
                    print(
                        'epoch: [%2d/%2d], batch: [%5d/%5d],' % (epoch, self.config.epoch, i, batch_num),
                        ' ||  loss4 : %10.4f  ||  loss5 : %10.4f' % (batch_avg_loss4, batch_avg_loss5)
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    aveGrad = 0
                    batch_avg_loss4 = 0  # 该mini-batch中每张图片的平均loss(最后1个side output的loss)
                    batch_avg_loss5 = 0  # 该mini-batch中每张图片的平均loss(所有side output的loss之和)

            # 每几个epoch保存一下模型权重
            if (epoch + 1) % self.config.epoch_save == 0:
                torch.save(self.net.state_dict(), '%s/models/epoch_%d.pth' % (self.config.save_folder, epoch))

            # 记录每个epoch的学习率
            self.tb_log(data=self.lr, step=epoch, mode='lr')

            # 对应epoch对学习率进行lr dacay
            if epoch in self.lr_decay_epoch:
                self.lr = self.lr * 0.1
                self.optimizer = Adam(filter(lambda p: p.requires_grad, self.net.parameters()), lr=self.lr, weight_decay=self.wd)

            # 每个epoch验证1下模型
            self.eval_model(epoch)

        torch.save(self.net.state_dict(), '%s/models/final.pth' % self.config.save_folder)
        self.logger.export_scalars_to_json('%s/tblog/all_scalars.json' % self.config.save_folder)
        print('best epoch: %d || best MAE: %10.4f || best F_measure: %10.4f' % (self.best_epoch, self.best_mae, self.best_F_measure))

    def eval_model(self, epoch):
        with torch.no_grad():
            # 验证模型
            mae = 0.0
            F_measure = 0.0
            for img, gt in self.eval_loader:
                preds = self.net(img)
                pred = torch.squeeze(preds[4])
                mae += evaluator.calc_MAE(pred, gt).item()
                img_p, img_r = evaluator.calc_precision_recall(pred, gt) # 返回的不是tensor
                F_measure += evaluator.calc_F_measure(img_p, img_r)# 返回的不是tensor
            mae /= len(self.eval_loader.dataset)
            F_measure /= len(self.eval_loader.dataset)
        # 更新
        if mae < self.best_mae:
            self.best_mae = mae
            self.best_F_measure = F_measure
            self.best_epoch = epoch
        elif self.best_mae == mae:
            if F_measure > self.best_F_measure:
                self.best_F_measure = F_measure
                self.best_epoch = epoch
        
        # 输出日志
        print('epoch: [ %2d/%2d] || mae : %10.4f || F_measure : %10.4f' %(epoch, self.config.epoch, mae, F_measure))
        self.logger.add_scalar('MAE', mae, epoch)
        self.logger.add_scalar('F_measure', F_measure, epoch)

    def tb_log(self, data, step, mode):
        if mode == 'layer&loss':  # 记录6个层的4种loss
            losses = data
            # Loss维度：有4张图，1个图代表1种loss，1个图中6条曲线分别是6个层的该loss
            for j in range(4):
                self.logger.add_scalars('LossDim/'+self.lossType[j], {
                    '0': losses[0][j],
                    '1': losses[1][j],
                    '2': losses[2][j],
                    '3': losses[3][j],
                    '4': losses[4][j],
                    '5': losses[5][j],
                }, step)
            # Layer维度：有6个图，1个图代表1个层，1个图中4条曲线分别是该层的4种loss
            for i in range(6):
                self.logger.add_scalars('LayerDim/%d' % i, {
                    self.lossType[0]: losses[i][0],
                    self.lossType[1]: losses[i][1],
                    self.lossType[2]: losses[i][2],
                    self.lossType[3]: losses[i][3],
                }, step)
        elif mode == 'lr':
            lr = data
            self.logger.add_scalar('LR', lr, step)
