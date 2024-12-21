
import cv2
import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
# from mask_rle import rle2mask
from torchvision import transforms
from PIL import Image

import albumentations as A
from albumentations.pytorch import ToTensorV2
# PATH = os.path.join('/', 'media', 'arthasjian', 'SharedFiles', '工作', 'DataSets', 'severstal-steel-defect-detection')
PATH = "/home/lcheng/code/dataset/severstal-steel-defect-detection"

class SteelDataset(Dataset):
    def __init__(self, train=True, transform=None):
        super().__init__()
        self.transform = transform
        self.train = train

        if self.train:
            self.csv_file = os.path.join(PATH, 'train.csv')
            self.df = pd.read_csv(self.csv_file)
            self.path = os.path.join(PATH, 'train_images')
            self.files = self.df['ImageId'].unique().tolist()
        else:
            self.path = os.path.join(PATH, 'test_images')

    def __getitem__(self, index):
        part_index = index % 8

        img_index = index // 8
        if self.train:
            imageId = self.files[img_index]
        else:
            imageId = os.listdir(self.path)[img_index]

        image_path = os.path.join(self.path, imageId)
        # image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = Image.open(image_path).convert('L')
        image = np.array(image)  # (256, 1600)
        matrix = np.zeros((4, image.shape[0], image.shape[1]), dtype=np.uint8)  # (4, 256, 1600)

        if self.train:
            rle = self.df.where(self.df['ImageId'] == imageId).dropna()
            for i in range(4):
                try:
                    RLE = rle.where(rle['ClassId'] == i + 1).dropna()['EncodedPixels'].iloc[0]
                    channel_label = rle2mask(RLE)
                    matrix[i] = channel_label
                except:
                    continue

            label = matrix[0] + (matrix[1] * 2) + (matrix[2] * 4) + (matrix[3] * 8)
            part_label = label[:, (part_index * 200):(part_index * 200 + 200)]

            while part_label.sum().item() == 0:
                part_index = (part_index + 1) % 8
                part_label = label[:, (part_index * 200):(part_index * 200 + 200)]

            part_image = image[:, (part_index * 200):(part_index * 200 + 200)]  # (256, 200)

            train_augmentations = A.Compose([
                A.HorizontalFlip(p=0.5),  # 50%随即水平翻转
                A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15,
                                   interpolation=cv2.INTER_LINEAR, p=0.5),
                # 以 50% 的概率对图像进行最大 10% 的平移、10% 的缩放和 ±15° 的旋转，并使用双线性插值处理像素值
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # 标准化
                ToTensorV2()
            ])
            augmented = train_augmentations(image=part_image)
            part_image = augmented['image']
            # totensor = transforms.ToTensor()
            # trans = self.transform

            # part_image = totensor(part_image)
            # part_image = trans(part_image)  # (1, 256, 200)
            part_label = torch.tensor(part_label)  # (256, 200)
            return part_image, part_label.long()

        else:
            part_image = image[:, (part_index * 200):(part_index * 200 + 200)]  # (256, 200)

            test_augmentations = A.Compose([
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), # 标准化
                ToTensorV2()
            ])
            augmented = test_augmentations(image=part_image)
            part_image = augmented['image']
            # totensor = transforms.ToTensor()
            # trans = self.transform
            # part_image = totensor(part_image)
            # part_image = trans(part_image)
            fake_y = torch.zeros([256, 200])
            return part_image, fake_y.long()

    def __len__(self):
        if self.train:
            return len(self.files) * 8
        else:
            return len(os.listdir(self.path)) * 8


def main():
    transform = transforms.Compose([
        transforms.Resize((256, 200)),
    ])
    dataset = SteelDataset(train=True, transform=transform)
    # tqdmbar = tqdm(range(8))
    for i in range(5):
        image, mask = dataset[i]
        from torchvision.transforms import ToPILImage
        image = image.permute(1, 2, 0).numpy()
        image = ToPILImage()(image)
        plt.imshow(image, cmap='gray')
        plt.show()



if __name__ == '__main__':
    main()
