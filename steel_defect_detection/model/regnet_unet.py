# -*- coding:utf-8 -*-
# Time: 2024/12/20
# Author: lcheng1334
# File: regnet-unet.py
import torch
import torch.nn as nn
from torchvision.models import regnet_y_400mf
import segmentation_models_pytorch as smp


class RegNetSMPUnet(nn.Module):
    def __init__(self, num_classes=2, encoder_weights='imagenet'):
        super(RegNetSMPUnet, self).__init__()

        # 使用 SMP 支持的 RegNet 编码器
        self.decoder = smp.Unet(
            encoder_name="timm-regnety_002",  # 替换为 SMP 支持的编码器
            encoder_weights=encoder_weights,  # 预训练权重
            in_channels=3,  # 输入图像通道数 (RGB)
            classes=num_classes  # 输出类别数
        )

    def forward(self, x):
        return self.decoder(x)


# 测试模型
if __name__ == "__main__":
    model = RegNetSMPUnet(num_classes=2, encoder_weights='imagenet')  # 二分类
    # print(model)
    x = torch.randn(1, 3, 256, 256)  # 假设输入大小为 256x256 的 RGB 图像
    output = model(x)
    print("Output shape:", output.shape)  # 应为 [1, 1, 256, 256]
