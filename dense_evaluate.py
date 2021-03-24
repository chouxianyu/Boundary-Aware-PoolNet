from torchvision import transforms
import os
import sys
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class EvalDataset(Dataset):
    def __init__(self, method, pred_dir_path, gt_dir_path, cuda):
        self.cuda = cuda
        self.trans = transforms.Compose([transforms.ToTensor()])
        if not os.path.exists(pred_dir_path) or not os.path.exists(gt_dir_path):
            print('ERROR : dir path not exist!')
        # 模型预测结果的路径
        if method=='CPD':
            self.pred_paths = [os.path.join(pred_dir_path, file_name) for file_name in sorted(os.listdir(
                pred_dir_path)) if not any(x in file_name for x in ['ILSVRC2012_test_00036002', 'sun_bcogaqperiljqupq'])]
        else:
            self.pred_paths = [os.path.join(pred_dir_path, file_name) for file_name in sorted(os.listdir(pred_dir_path))]
        # 图片标注的路径
        if method=='CapSal' or method=='CPD':  # (需要通过if来处理CapSal只有5017张预测图的情况和CPD未处理测试集中2张错误图片的情况)
            self.gt_paths = [os.path.join(gt_dir_path, file_name) for file_name in sorted(os.listdir(
                gt_dir_path)) if not any(x in file_name for x in ['ILSVRC2012_test_00036002', 'sun_bcogaqperiljqupq'])]
        else:
            self.gt_paths = [os.path.join(gt_dir_path, file_name) for file_name in sorted(os.listdir(gt_dir_path))]
            
        # 图片数量
        self.img_num = len(self.gt_paths)
        if len(self.pred_paths) != len(self.gt_paths):
            print('ERROR : file number not equal')
            print("Pred Num", len(self.pred_paths))
            print("GT Num", len(self.gt_paths))
            sys.exit(1)

    def __getitem__(self, index: int):
        
        gt = self.trans(Image.open(self.gt_paths[index]).convert('L'))
        pred = np.array(Image.open(self.pred_paths[index]).convert('L'))
        pred = ((pred - pred.min()) / (pred.max() - pred.min()) * 255).astype(np.uint8)
        pred = self.trans(Image.fromarray(pred))
        if self.cuda:
            return pred.cuda(), gt.cuda()
        return pred, gt

    def __len__(self):
        return self.img_num


class Evaluator:
    def __init__(self, method, thresholds, pred_dir_path, output_path, cuda=True, gt_dir_path='./data/DUTS/DUTS-TE/DUTS-TE-Mask'):
        self.method = method
        self.loader = EvalDataset(method, pred_dir_path, gt_dir_path, cuda)
        self.cuda = cuda
        self.thresholds = thresholds
        self.threshold_cnt = len(self.thresholds)
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        self.output_path = output_path
        self.img_num = self.loader.img_num
        # metric data
        self.mae = 0.0
        self.max_F_measure = 0.0
        self.precisions = torch.zeros(self.threshold_cnt)
        self.recalls = torch.zeros(self.threshold_cnt)
        self.F_measures = torch.zeros(self.threshold_cnt)
        # 计算并保存metrics
        self.run()

    def _calc_MAE(self):
        print('MAE Calculating……')
        overall_mae = 0.0 # 整个数据集图片的MAE=所有图片MAE的均值
        img_num = 0.0
        with torch.no_grad():
            for pred, gt in self.loader:
                img_mae = torch.abs(pred - gt).mean() # 一张图片的MAE
                if img_mae == img_mae:  # for Nan
                    overall_mae += img_mae
                    img_num += 1.0
                else:
                    print('nan!')
            overall_mae /= img_num
            self.mae = overall_mae.item()  # 整个数据集图片的MAE=所有图片MAE的均值
            print('MAE Calculated!')
            print('Valid img num:', img_num)
            print('MAE: ', self.mae)

    def _calc_F_measures(self, beta2=0.3):
        print('F_measures Calculating……')
        with torch.no_grad():
            for i in range(self.threshold_cnt):  # 1个阈值对应1组PR
                threshold = self.thresholds[i]
                overall_p = 0  # 1个数据集的PR为数据集中所有图片PR的均值
                overall_r = 0  # 1个数据集的PR为数据集中所有图片PR的均值
                for pred, gt in self.loader:
                    p, r = self._calc_pr(pred, gt, threshold)
                    # print('precision', p, 'recall', r)
                    overall_p += p
                    overall_r += r
                overall_p /= self.img_num # 1个数据集的PR为数据集中所有图片PR的均值
                overall_r /= self.img_num # 1个数据集的PR为数据集中所有图片PR的均值
                self.precisions[i] = overall_p
                self.recalls[i] = overall_r
                self.F_measures[i] = (1+beta2)*overall_p*overall_r / (beta2*overall_p+overall_r)
                print('Threshold: %f || Precision: %10.4f || Recall: %10.4f || F-measure: %10.4f'
                    % (threshold*255, self.precisions[i], self.recalls[i], self.F_measures[i])
                )
            self.F_measures[self.F_measures != self.F_measures] = 0  # for Nan
            self.max_F_measure = self.F_measures.max().item()
            print('F_measures Calculated!')
            print('max_F_measure', self.max_F_measure)

    def _calc_pr(self, pred, gt, threshold):
        with torch.no_grad():
            pred = pred.ge(threshold)
            gt = gt.ge(0.5)
            TP = torch.sum(pred*gt)
            p = (TP+ 1e-20)/(pred.sum()+1e-20)
            r = (TP+ 1e-20)/(gt.sum()+1e-20)
            return p, r

    def _save(self):
        with torch.no_grad():
            ## tensor转numpy array
            if self.cuda:
                F_measures = self.F_measures.cpu().numpy()
                precisions = self.precisions.cpu().numpy()
                recalls = self.recalls.cpu().numpy()
            else:
                F_measures = self.F_measures.numpy()
                precisions = self.precisions.numpy()
                recalls = self.recalls.numpy()
            ## 保存数据
            np.savetxt(os.path.join(self.output_path,'F_measures.csv'), F_measures, delimiter=',')
            print('F_measures.csv Saved')
            np.savetxt(os.path.join(self.output_path, 'precisions.csv'), precisions, delimiter=',')
            print('precisions.csv Saved')
            np.savetxt(os.path.join(self.output_path, 'recalls.csv'), recalls, delimiter=',')
            print('recalls.csv Saved')
            with open(os.path.join(self.output_path, 'mae.txt'), 'w') as f:
                f.write(str(self.mae))
                f.close()
                print('mae.txt Saved')

            with open(os.path.join(self.output_path, 'max_F_measure.txt'), 'w') as f:
                f.write(str(self.max_F_measure))
                f.close()
                print('max_F_measure.txt Saved')

    def run(self):
        print('='*50)
        print('Evaluating Method:', self.method)
        print('='*50)
        print()
        self._calc_MAE()
        print()
        self._calc_F_measures()
        print()
        self._save()
        print()
        print('='*50)
        print('Method: %s evaluated, results saved in %s' % (self.method, self.output_path) )
        print('MAE: %10.4f || Max F-measure: %10.4f' % (self.mae, self.max_F_measure) )
        print('='*50)
        print()
        print()
        print()


def main():
    method = 'BAPoolNet'
    # config
    thresholds = np.linspace(0, 1, 256)
    os.environ['CUDA_VISIBLE_DEVICES'] = '6'
    pred_dir_path = './results/run-5/test_epoch20'
    output_path = './eval_results/run3_epoch20'

    # eval
    Evaluator(method, thresholds, pred_dir_path, output_path)


def multi_main():
    # config
    methods = ['CapSal', 'PiCANet', 'DGRL', 'BASNet', 'U2Net', 'CPD', 'PoolNet', 'BAPoolNet']
    thresholds = np.linspace(0, 1, 256)
    os.environ['CUDA_VISIBLE_DEVICES'] = '6'

    # eval
    for method in methods:
        pred_dir_path = os.path.join('./data/predictions/', method)
        output_path = os.path.join('./eval_results', method)
        Evaluator(method, thresholds, pred_dir_path, output_path)


if __name__ == '__main__':
    # main()
    multi_main()
