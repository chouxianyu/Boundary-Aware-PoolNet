import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from dataset.dataset import load_image_test
from networks.dense_poolnet import build_model


def show(img_name, img, pred):
    #convert BGR to RGB
    img = img[:, :, ::-1]
    # img
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title(img_name)
    plt.xticks([])
    plt.yticks([])
    # pred
    plt.subplot(1, 2, 2)
    plt.imshow(pred, cmap='gray')
    plt.title(img_name+'_pred')
    plt.xticks([])
    plt.yticks([])
    plt.show()


def main():
    # config
    img_dir_path = './examples'
    output_dir_path = './outputs'
    weight_path = './scratch_results/run-5/models/epoch_20.pth'
    cuda = False
    show_every = True

    # inference
    img_names = os.listdir(img_dir_path)
    img_num = len(img_names)
    net = build_model('resnet')  # 构建PoolNet网络结构
    if cuda:
        net = net.cuda()
    net.eval()
    if cuda:
        net.load_state_dict(torch.load(weight_path))
    else:
        net.load_state_dict(torch.load(weight_path, 'cpu'))
    for i in range(img_num):
        img_name = img_names[i]
        image, _ = load_image_test(os.path.join(img_dir_path, img_name))
        image = image[np.newaxis, ...]  # (C,H,W) => (1,C,H,W)
        with torch.no_grad():
            image = torch.from_numpy(image)
            if cuda:
                image = image.cuda()
            preds = net(image)
            pred = np.squeeze(preds[4].cpu().data.numpy())
            pred = 255 * pred
            cv2.imwrite(os.path.join(output_dir_path,img_name[:-4] + '_' + 'inference' + '.png'), pred)
        if show_every:
            img = cv2.imread(os.path.join(img_dir_path, img_name))
            show(img_name[:-4], img, pred)
        print(i, 'Inferenced', img_name)


if __name__ == '__main__':
    main()
