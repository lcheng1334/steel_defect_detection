# _*_ coding: utf-8 _*_
#
# Copyright (C) 2024 - 2024 Zhijian Zhu, All Rights Reserved 
#
# @Time    : 2024/12/9 下午8:14
# @Author  : Zhijian Zhu
# @File    : test.py
# @IDE     : PyCharm
import matplotlib as mpl
# mpl.rcParams['font.sans-serif'] = ['SimHei']
# mpl.rcParams['axes.unicode_minus'] = False
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "data_preprocessing"))

from matplotlib import pyplot as plt

from data_preprocessing import SteelDataset
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
from model import UNet
from train_and_eval import TrainAndEvalLightning


def plot_result(xy_list: list, model: nn.Module):
    mapped_x = torch.cat([xy_list[i][0] for i in range(8)], dim=2)
    batch_x = torch.cat([xy_list[i][0].reshape(-1, 1, 256, 200) for i in range(8)], dim=0)
    mapped_y = torch.cat([xy_list[i][1] for i in range(8)], dim=1)

    batch_y_pred = model(batch_x)
    batch_y_pred = torch.argmax(batch_y_pred, dim=1)

    pred_image = torch.zeros([256, 1600])
    for i in range(8):
        pred_image[:, i * 200:(i + 1) * 200] = batch_y_pred[i, :, :]

    fig, ax = plt.subplots(nrows=3, ncols=1)

    ax[0].imshow(mapped_x.cpu().detach().numpy().transpose(1, 2, 0), cmap='gray')
    # ax[0].set_title('origin')
    ax[0].axis('off')

    ax[2].imshow(pred_image.cpu().detach().numpy(), cmap='gray')
    # ax[2].set_title('predict')
    ax[2].axis('off')

    ax[1].imshow(mapped_x.cpu().detach().numpy().transpose(1, 2, 0), cmap='gray')
    ax[1].imshow(pred_image.cpu().detach().numpy(), alpha=0.5, cmap='Grays')
    # ax[1].set_title('origin+predict')
    ax[1].axis('off')

    # ax[3].imshow(mapped_y.cpu().detach().numpy(), cmap='Grays')
    # # ax[3].set_title('label')
    # ax[3].axis('off')
    # plt.savefig("end.png", bbox_inches='tight', pad_inches=0)
    plt.show()

def main():
    transform = transforms.Compose([
        transforms.Resize([256, 200])
    ])
    train_dataset = SteelDataset(train=True, transform=transform)

    model = TrainAndEvalLightning.load_from_checkpoint('./logs/medium_precision/version_0/checkpoints/epoch=2-step=8001.ckpt',
                                                       model=UNet(1, 16), lr=0.01)



    # for img in range(10):
    #     xy_list = []
    #     for i in range(img * 8, (img + 1) * 8):
    #         x, y = train_dataset[i]  # x(1, 256, 200) y(256, 200)
    #         xy_list.append((x, y))
    #
    #     plot_result(xy_list, model_sum)


    import cv2
    import os
    from torchvision.transforms import ToTensor

    from data_preprocessing import rle2mask, mask2rle

    path = '/home/lc/code/datasets/severstal-steel-defect-detection/train_images'
    image_path = sorted(os.listdir(path))[2]
    print(image_path)
    img = cv2.imread(os.path.join(path, image_path), cv2.IMREAD_GRAYSCALE)

    img = ToTensor()(img)
    print(img.shape)
    label = torch.zeros(8, 256, 200)
    xy_list = []
    for i in range(8):
        xy_list.append((img[0, :, i * 200:(i + 1) * 200].reshape(-1, 256, 200), label[i]))

    plot_result(xy_list, model)











if __name__ == '__main__':
    main()