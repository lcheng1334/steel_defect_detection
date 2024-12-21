# -*- coding:utf-8 -*-
# Time: 2024/12/21
# Author: lcheng1334
# File: preprocessing.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torch import nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
import warnings
warnings.filterwarnings("ignore")
from torch.utils.data import Dataset
from torchvision.transforms import ToPILImage
import cv2

def mask2rle(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels= img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def rle2mask(mask_rle, shape=(1600,256)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (width,height) of array to return
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T


class SteelDefectDataset(Dataset):
    def __init__(self, root_dir, train: bool, num_parts: int = 5):
        super(SteelDefectDataset, self).__init__()
        self.root_dir = root_dir
        self.train = train
        self.num_parts = num_parts

        if self.train:
            self.csv = os.path.join(root_dir, 'train.csv')
            self.df = pd.read_csv(self.csv)
            self.path = os.path.join(self.root_dir, 'train_images')
            self.train_files = sorted(self.df['ImageId'].unique().tolist())
        else:
            self.path = os.path.join(self.root_dir, 'test_images')
            self.test_files = sorted(os.listdir(self.path))

    def __getitem__(self, index):
        part_index = index % self.num_parts  # 使用参数化分块数量
        img_index = index // self.num_parts

        if self.train:
            imageId = self.train_files[img_index]
        else:
            imageId = self.test_files[img_index]

        image_path = os.path.join(self.path, imageId)
        image = Image.open(image_path).convert('RGB')  # 直接读取原图（RGB）
        image = np.array(image)  # 假设原图为 (256, 1600, 3)

        if self.train:
            rle = self.df.where(self.df['ImageId'] == imageId).dropna()
            try:
                RLE = rle['EncodedPixels'].iloc[0]
                label_mask = rle2mask(RLE)  # 直接生成单通道掩码
            except:
                label_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)  # 如果没有缺陷，则为全零掩码

            part_label = label_mask[:, (part_index * 320):(part_index * 320 + 320)]
            part_image = image[:, (part_index * 320):(part_index * 320 + 320), :]  # 提取分块图像

            # 跳过全零掩码
            if part_label.sum() == 0:
                new_index = (index + 1) % len(self)  # 防止越界，循环处理
                return self.__getitem__(new_index)

            # 图像和掩码同步增强
            augmentations = A.Compose([
                A.Resize(256, 256),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # 仅对图像标准化
                ToTensorV2()
            ])
            augmented = augmentations(image=part_image, mask=part_label)
            part_image = augmented['image']  # 图像增强和标准化
            part_label = augmented['mask']  # 掩码同步增强

            return part_image, part_label

        else:
            part_image = image[:, (part_index * 320):(part_index * 320 + 320), :]

            augmentations = A.Compose([
                A.Resize(256, 256),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])
            augmented = augmentations(image=part_image)
            part_image = augmented['image']

            return part_image

    def __len__(self):
        if self.train:
            return len(self.train_files) * self.num_parts  # 使用参数化分块数量
        else:
            return len(self.test_files) * self.num_parts

def main():
    PATH = "/home/lcheng/code/dataset/severstal-steel-defect-detection"
    train_data = SteelDefectDataset(root_dir=PATH, train=True, num_parts=5)

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    for i in range(6):
        image, label = train_data[i]

        # 转换张量为 NumPy 格式
        image = image.permute(1, 2, 0).numpy()  # 转置为 (H, W, C)
        label = label.numpy()  # 转换为 NumPy 数组

        # 反归一化图像
        image = std * image + mean  # 恢复原始像素值范围
        image = np.clip(image, 0, 1)  # 确保范围在 [0, 1]

        # 显示图像和掩码
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title("Image")
        plt.imshow(image)  # 显示 RGB 图像
        plt.subplot(1, 2, 2)
        plt.title("Label")
        plt.imshow(label, cmap='gray')  # 显示掩码
        plt.show()

if __name__ == '__main__':
    main()

