from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import os
import cv2
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
from dataset.dataset import load_image_test
from networks.dense_poolnet import build_model, PoolNet
from PyQt5.QtWidgets import QWidget, QApplication, QGridLayout, QVBoxLayout,QPushButton, QLabel, QFrame, QFileDialog, QLineEdit
from PyQt5.QtCore import Qt


class SingleModeWindow(QWidget):
    def __init__(self, model: PoolNet, cuda:bool):
        # 基类初始化
        super().__init__()

        # 数据初始化
        self.img_path_line_edit = QLineEdit()
        self.model = model
        self.cuda = cuda
        self.fig = Figure(dpi=200)
        self.canvas = FigureCanvas(self.fig)

        # UI初始化
        self.init_ui()
    
    def init_ui(self):
        # 设置窗口标题
        self.setWindowTitle('单幅模式')

        # 生成网格布局
        main_layout = QGridLayout()
        ## 第一行
        self.img_path_line_edit.setFocusPolicy(Qt.NoFocus)  # 设置不可编辑
        main_layout.addWidget(self.img_path_line_edit, 0, 0, 1, 5)
        img_path_btn = QPushButton('选择图片')
        img_path_btn.setStyleSheet("QPushButton{font:20px;}")
        img_path_btn.clicked.connect(self.on_click_img_path_btn)
        main_layout.addWidget(img_path_btn, 0, 5, 1, 1)
        ## 第三行
        main_layout.addWidget(self.canvas, 1, 0, 2, 6)

        # 设置布局
        self.setLayout(main_layout)

        #设置窗口大小
        self.resize(1400, 800)

    def on_click_img_path_btn(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选取图片", os.getcwd(), "JPEG Files (*.jpg)")
        if file_path != '':
            self.img_path_line_edit.setText(file_path)
            self.update_canvas()

    def update_canvas(self):
        # 图片路径
        img_path = self.img_path_line_edit.text()

        # Inference
        print('Inferencing ……')
        image, _ = load_image_test(img_path)
        image = image[np.newaxis, ...]  # (C,H,W) => (1,C,H,W)
        with torch.no_grad():
            image = torch.from_numpy(image)
            if self.cuda:
                image = image.cuda()
            preds = self.model(image)
        
        # Plot
        print('PLotting ……')
        ## 原图
        ax = self.fig.add_subplot(1, 6, 1)
        ax.cla() # 清空画布
        ax.imshow(cv2.imread(img_path)[:, :, ::-1])  # convert BGR to RGB
        ax.set_title('原图')
        ax.set_xticks([])
        ax.set_yticks([])
        ## BAPoolNet的5个输出
        for i in range(5):
            pred = np.squeeze(preds[i].cpu().data.numpy())
            pred = 255 * pred
            ax = self.fig.add_subplot(1, 6, i+2)
            ax.cla()  # 清空画布
            ax.imshow(pred, cmap='gray')
            ax.set_title('Output' + str(i+1))
            ax.set_xticks([])
            ax.set_yticks([])
        self.canvas.draw()


class BatchModeWindow(QWidget):
    def __init__(self, model: PoolNet, cuda: bool):
        # 基类初始化
        super().__init__()

        # 数据初始化
        self.input_dir_path_line_edit = QLineEdit('./examples')
        self.output_dir_path_line_edit = QLineEdit('./outputs')
        self.model = model
        self.cuda = cuda

        # UI初始化
        self.init_ui()

    def init_ui(self):
        # 设置窗口标题
        self.setWindowTitle('批量模式')

        # 生成网格布局
        main_layout = QVBoxLayout()
        ## 第一行
        input_dir_path_btn = QPushButton('输入文件夹')
        input_dir_path_btn.setStyleSheet("QPushButton{font:20px;}")
        input_dir_path_btn.clicked.connect(self.on_click_input_dir_path_btn)
        main_layout.addWidget(input_dir_path_btn)
        ## 第二行
        self.input_dir_path_line_edit.setFocusPolicy(Qt.NoFocus)  # 设置不可编辑
        main_layout.addWidget(self.input_dir_path_line_edit)
        ## 第三行
        output_dir_path_btn = QPushButton('输出文件夹')
        output_dir_path_btn.setStyleSheet("QPushButton{font:20px;}")
        output_dir_path_btn.clicked.connect(self.on_click_output_dir_path_btn)
        main_layout.addWidget(output_dir_path_btn)
        ## 第四行
        self.output_dir_path_line_edit.setFocusPolicy(Qt.NoFocus)  # 设置不可编辑
        main_layout.addWidget(self.output_dir_path_line_edit)
        ## 第五行
        start_inference_btn = QPushButton('开始推理')
        start_inference_btn.setStyleSheet("QPushButton{font:20px;}")
        start_inference_btn.clicked.connect(self.on_click_start_inference_btn)
        main_layout.addWidget(start_inference_btn)
        
        # 设置布局
        self.setLayout(main_layout)
        
        #设置窗口大小
        self.resize(800, 200)


    def on_click_input_dir_path_btn(self):
        dir_path = QFileDialog.getExistingDirectory(self, "选取输入文件夹", os.getcwd())
        if dir_path != '':
            self.input_dir_path_line_edit.setText(dir_path)

    def on_click_output_dir_path_btn(self):
        dir_path = QFileDialog.getExistingDirectory(self, "选取输出文件夹", os.getcwd())
        if dir_path != '':
            self.output_dir_path_line_edit.setText(dir_path)

    def on_click_start_inference_btn(self):
        # 处理文件和文件夹
        input_dir_path = self.input_dir_path_line_edit.text()
        output_dir_path = self.output_dir_path_line_edit.text()
        img_names = os.listdir(input_dir_path)
        img_num = len(img_names)
        print('Load Images From %s' % input_dir_path)
        print('Output Images To %s' % output_dir_path)
        print('Image Num:', img_num)

        # Inference
        for i in range(img_num):
            img_name = img_names[i]
            image, _ = load_image_test(os.path.join(input_dir_path, img_name))
            image = image[np.newaxis, ...]  # (C,H,W) => (1,C,H,W)
            with torch.no_grad():
                image = torch.from_numpy(image)
                if self.cuda:
                    image = image.cuda()
                preds = self.model(image)
                pred = np.squeeze(preds[4].cpu().data.numpy())
                pred = 255 * pred
                cv2.imwrite(os.path.join(output_dir_path, img_name[:-4] + '_' + 'inference' + '.png'), pred)
            print(i+1, 'Inferenced', img_name)


class MainWindow(QWidget):
    def __init__(self, cuda=False):
        # 基类初始化
        super().__init__()

        # 数据初始化
        self.cuda = cuda

        self.weight_path_line_edit = QLineEdit('./results/run-5/models/epoch_20.pth')
        self.model = build_model()  # 构建PoolNet网络结构
        if self.cuda:
            self.model = self.model.cuda()
        self.model.eval()
        if self.cuda:
            self.model.load_state_dict(torch.load(self.weight_path_line_edit.text()))
        else:
            self.model.load_state_dict(torch.load(self.weight_path_line_edit.text(), 'cpu'))
        print('Load Weights From %s' % self.weight_path_line_edit.text())
        self.single_mode_window = SingleModeWindow(self.model, self.cuda)
        self.batch_mode_window = BatchModeWindow(self.model, self.cuda)

        # UI初始化
        self.init_ui()
        
        # 显示窗口
        self.show()

    def init_ui(self):
        # 设置窗口标题
        self.setWindowTitle('Boundary Aware PoolNet')

        # 生成网格布局
        main_layout = QGridLayout()

        # 在界面中添加控件
        ## 第一行
        title_label = QLabel('<font size="20">显著性目标检测系统</font>')
        title_label.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label, 0, 0, 1, 3)
        ## 第二行
        self.weight_path_line_edit.setFocusPolicy(Qt.NoFocus)  # 设置不可编辑
        main_layout.addWidget(self.weight_path_line_edit, 1, 0, 1, 2)
        weight_path_btn = QPushButton('选择权重文件')
        weight_path_btn.setStyleSheet("QPushButton{font:20px;}")
        weight_path_btn.clicked.connect(self.on_click_weight_path_btn)
        main_layout.addWidget(weight_path_btn, 1, 2, 1, 1)
        ## 第三行
        single_mode_btn = QPushButton('单幅模式')
        single_mode_btn.setStyleSheet("QPushButton{font:40px;}")
        single_mode_btn.clicked.connect(self.single_mode_window.show)
        main_layout.addWidget(single_mode_btn, 2, 0, 1, 3)
        ## 第四行
        batch_mode_btn = QPushButton('批量模式')
        batch_mode_btn.setStyleSheet("QPushButton{font:40px;}")
        batch_mode_btn.clicked.connect(self.batch_mode_window.show)
        main_layout.addWidget(batch_mode_btn, 3, 0, 1, 3)

        # 设置布局
        self.setLayout(main_layout)

        # 设置窗口大小
        self.resize(600, 400)

    def on_click_weight_path_btn(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选取文件", os.getcwd(), "PyTorch Weights(*.pth)")
        if file_path != '':
            self.weight_path_line_edit.setText(file_path)
            if self.cuda:
                self.model.load_state_dict(torch.load(self.weight_path_line_edit.text()))
            else:
                self.model.load_state_dict(torch.load(self.weight_path_line_edit.text(), 'cpu'))
            print('Load Weight From %s' % self.weight_path_line_edit.text())


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    sys.exit(app.exec_())
