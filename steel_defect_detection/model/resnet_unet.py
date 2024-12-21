# -*- coding:utf-8 -*-
# Time: 2024/12/20
# Author: lcheng1334
# File: resnet-unet.py
import segmentation_models_pytorch as smp
import torch
from torch import nn

class ResNetSMPUNet(nn.Module):
    def __init__(self, num_classes=2, encoder_weights="imagenet"):
        super(ResNetSMPUNet, self).__init__()
        self.decoder = smp.Unet(
            encoder_name="resnet34",
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=num_classes
        )
    def forward(self, x):
        return self.decoder(x)
def main():
    x = torch.randn(1, 3, 256, 256)
    model = ResNetSMPUNet()
    output = model(x)
    print(output.shape)

if __name__ == '__main__':
    main()