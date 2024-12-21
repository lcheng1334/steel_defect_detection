# -*- coding:utf-8 -*-
# Time: 2024/12/21
# Author: lcheng1334
# File: main_update.py

import os
from data_preprocessing import SteelDefectDataset
from train_and_eval import LightningModel
import torch
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping  # 回调函数：模型保存和早停
from pytorch_lightning import Trainer
from model import UNet, ResNetSMPUNet, RegNetSMPUnet

def train_all_model(models, train_dataloader, val_dataloader, test_dataloader):
    for model_name, model in models.items():
        print(f'Training {model_name}')

        # 初始化 Lightning 模型
        lightning_model = LightningModel(model=model, lr=1e-3)

        # 定义 TensorBoard 日志记录器
        logger = TensorBoardLogger('logs', name=model_name)

        # 定义 Trainer
        trainer = Trainer(
            max_epochs=50,
            logger=logger,
            callbacks=[
                ModelCheckpoint(
                    monitor='val_loss',  # 监控验证集损失
                    dirpath=f'checkpoints/{model_name}',
                    filename="{epoch}-{val_loss:.2f}",
                    save_top_k=1,  # 仅保存验证集损失最低的模型
                    mode='min',  # 目标是最小化验证集损失
                ),
                EarlyStopping(
                    monitor='val_loss',  # 监控验证集损失
                    patience=5,  # 如果验证集损失 5 个周期未改善，则停止训练
                    mode='min',  # 目标是最小化验证集损失
                    verbose=True,  # 输出早停信息
                )
            ]
        )
        # 开始训练
        trainer.fit(lightning_model, train_dataloader, val_dataloader)
        # 测试模型
        trainer.test(lightning_model, test_dataloader)

def main():

    PATH = "/home/lcheng/code/dataset/severstal-steel-defect-detection"

    # 加载数据集
    train_val_dataset = SteelDefectDataset(root_dir=PATH, train=True)
    test_dataset = SteelDefectDataset(root_dir=PATH, train=False)

    # 固定随机种子以确保数据划分一致
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(train_val_dataset, [0.8, 0.2], generator=generator)

    # 定义数据加载器
    batch_size = 8  # 每批数据的样本数
    num_workers = os.cpu_count()  # 使用 CPU 核心数
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)  # 训练集加载器
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)  # 验证集加载器
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)  # 测试集加载器

    # 定义模型
    models = {
        "unet": UNet(n_channels=3, n_classes=2),
        "resnet_unet": ResNetSMPUNet(num_classes=2, encoder_weights="imagenet"),
        "regnet_unet": RegNetSMPUnet(num_classes=2, encoder_weights="imagenet"),
    }


    train_all_model(models, train_dataloader, val_dataloader, test_dataloader)

if __name__ == '__main__':
    main()  # 运行主函数
