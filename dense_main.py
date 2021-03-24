import argparse
import os
from dataset.dataset import get_loader
from dataset import evaluator
from dense_solver import Solver
from torch.utils import data


def get_test_info(test_mode='t'):
    """
        根据test_mode，返回测试集图片根目录的路径以及测试集图片列表文件的路径
    """
    image_root = ''
    image_source = ''
    if test_mode == 'e':
        image_root = './data/ECSSD/Imgs/'
        image_source = './data/ECSSD/test.lst'
    elif test_mode == 'p':
        image_root = './data/PASCALS/Imgs/'
        image_source = './data/PASCALS/test.lst'
    elif test_mode == 'd':
        image_root = './data/DUTOMRON/Imgs/'
        image_source = './data/DUTOMRON/test.lst'
    elif test_mode == 'h':
        image_root = './data/HKU-IS/Imgs/'
        image_source = './data/HKU-IS/test.lst'
    elif test_mode == 's':
        image_root = './data/SOD/Imgs/'
        image_source = './data/SOD/test.lst'
    elif test_mode == 't':
        image_root = './data/DUTS/DUTS-TE/DUTS-TE-Image/'
        image_source = './data/DUTS/DUTS-TE/test.lst'
    elif test_mode == 'm_r': # for speed test
        image_root = './data/MSRA/Imgs_resized/'
        image_source = './data/MSRA/test_resized.lst'
    elif test_mode == 'pool_demo':
        image_root = './data/DUTS/DUTS-TE/DUTS-TE-Image/'
        image_source = './data/DUTS/DUTS-TE/poolnet_demo.lst'
    elif test_mode == 't500':
        image_root = './data/DUTS/DUTS-TE/DUTS-TE-Image/'
        image_source = './data/DUTS/DUTS-TE/test500.lst'
    
    return image_root, image_source


def main(config):
    if config.mode == 'train':
        train_loader = get_loader(config) # 加载训练集
        run = 0 # 为代码的本次运行生成id
        while os.path.exists("%s/run-%d" % (config.save_folder, run)):
            run += 1
        os.mkdir("%s/run-%d" % (config.save_folder, run)) # 创建保存该次运行结果的文件夹
        os.mkdir("%s/run-%d/models" % (config.save_folder, run)) # 创建保存该次运行所得模型文件的文件夹
        os.mkdir("%s/run-%d/tblog" % (config.save_folder, run)) # 创建保存该次训练日志的文件夹
        config.save_folder = "%s/run-%d" % (config.save_folder, run) # 该次运行结果的文件夹路径
        
        eval_loader = None
        if config.eval_epoch:
            eval_dataset = evaluator.EvalDataset('./data/DUTS/DUTS-TE/DUTS-TE-Image', './data/DUTS/DUTS-TE/DUTS-TE-Mask', config.cuda)
            eval_loader = data.DataLoader(dataset=eval_dataset, batch_size=config.batch_size, shuffle=False)

        train = Solver(train_loader, None, eval_loader, config)
        train.train()
    elif config.mode == 'test':
        test_loader = get_loader(config, mode='test')
        if not os.path.exists(config.test_fold): os.mkdir(config.test_fold) # 创建保存测试结果文件夹
        test = Solver(None, test_loader, None, config)
        if 'test01234' in config.test_fold:
            test.test([0, 1, 2, 3, 4])
        else:
            test.test([4])
    else:
        raise IOError("illegal input!!!")


if __name__ == '__main__':

    # backbone预训练模型文件路径
    resnet_path = './dataset/pretrained/resnet50_caffe.pth'

    parser = argparse.ArgumentParser()

    # Hyper-parameters
    parser.add_argument('--n_color', type=int, default=3) # 输入图片的通道数
    parser.add_argument('--lr', type=float, default=5e-5) # Learning rate
    parser.add_argument('--wd', type=float, default=0.0005) # Weight decay
    parser.add_argument('--no-cuda', dest='cuda', action='store_false') # store_false则默认使用CUDA，store_true则默认不使用CUDA
    os.environ['CUDA_VISIBLE_DEVICES'] = '6'
    # Training settings
    parser.add_argument('--arch', type=str, default='resnet') # 选择backbone
    parser.add_argument('--pretrained_model', type=str, default=resnet_path) # backbone预训练模型文件路径
    parser.add_argument('--epoch', type=int, default=24) # epoch数量
    parser.add_argument('--batch_size', type=int, default=1) # only support 1 now
    parser.add_argument('--num_thread', type=int, default=1) # 加载数据集时的线程数量
    parser.add_argument('--load', type=str, default='') # PoolNet模型加载路径，如果为''则只加载backbone预训练模型
    parser.add_argument('--save_folder', type=str, default='./ending_results') # 运行结果保存路径
    parser.add_argument('--epoch_save', type=int, default=1) # 模型文件保存间隔
    parser.add_argument('--iter_size', type=int, default=10) # 因为batch_size为1，所以iter_size为10就等同于batch_size为10
    parser.add_argument('--eval_epoch', type=bool, default=True)

    # Train data
    parser.add_argument('--train_root', type=str, default='./data/DUTS/DUTS-TR/') # 训练集图片根目录路径
    parser.add_argument('--train_list', type=str, default='./data/DUTS/DUTS-TR/train_pair.lst') # 训练集图片和标注列表文件的路径

    # Testing settings
    parser.add_argument('--model', type=str, default='./results/run-5/models/epoch_20.pth') # 待测试的模型文件的路径
    parser.add_argument('--test_fold', type=str, default='./results/run-5/test_epoch20') # 测试结果保存路径（包含test024即则输出3个通道，包含test01234则输出5个通道）
    parser.add_argument('--test_mode', type=str, default='t')  # 选择使用哪个测试集（t代表整个DUTS-TE，t500代表只选前500张图）

    # Misc
    parser.add_argument('--mode', type=str, default='test', choices=['train', 'test']) # 运行模式
    config = parser.parse_args()

    if not os.path.exists(config.save_folder): # 如果save_folder不存在则创建该文件夹
        os.mkdir(config.save_folder)

    # 获取测试集图片根目录的路径以及测试集图片列表文件的路径，训练和测试的时候都可以使用
    config.test_root, config.test_list = get_test_info(config.test_mode)

    print(config)
    main(config)
