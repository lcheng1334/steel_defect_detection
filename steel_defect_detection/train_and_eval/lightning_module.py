# -*- coding: utf-8 -*-

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics import Accuracy, MeanMetric, JaccardIndex, F1Score

class TrainAndEvalLightning(pl.LightningModule):
    def __init__(self, model: nn.Module, lr: float, num_classes: int):
        super().__init__()
        self.model = model
        self.lr = lr
        self.num_classes = num_classes

        # 使用 F1 Score 作为损失函数的实现
        self.criterion = F1Score(task='multiclass', num_classes=num_classes, average='macro')

        # 评价指标
        self.acc_1 = Accuracy(task='multiclass', num_classes=num_classes, top_k=1)
        self.acc_5 = Accuracy(task='multiclass', num_classes=num_classes, top_k=5)
        self.iou = JaccardIndex(task='multiclass', num_classes=num_classes)
        self.batch_loss = MeanMetric()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
        x, y = batch
        pred = self(x)

        # F1 Score Loss
        loss = 1 - self.criterion(pred, y)

        # 更新评价指标
        self.acc_1.update(pred, y)
        self.acc_5.update(pred, y)
        self.iou.update(pred, y)
        self.batch_loss.update(loss)

        # 记录日志
        self.log('train/loss', self.batch_loss.compute(), prog_bar=True)
        self.log('train/top-1_acc', self.acc_1.compute(), prog_bar=True)
        self.log('train/top-5_acc', self.acc_5.compute(), prog_bar=True)
        self.log('train/iou', self.iou.compute(), prog_bar=True)

        return loss

    def validation_step(self, batch):
        x, y = batch
        pred = self(x)

        # F1 Score Loss
        loss = 1 - self.criterion(pred, y)

        # 更新评价指标
        self.acc_1.update(pred, y)
        self.acc_3.update(pred, y)
        self.iou.update(pred, y)
        self.batch_loss.update(loss)

        # 记录日志
        self.log('valid/loss', self.batch_loss.compute(), prog_bar=True)
        self.log('valid/top-1_acc', self.acc_1.compute(), prog_bar=True)
        self.log('valid/top-5_acc', self.acc_5.compute(), prog_bar=True)
        self.log('valid/iou', self.iou.compute(), prog_bar=True)

        return loss

    def on_train_epoch_end(self):
        self.acc_1.reset()
        self.acc_5.reset()
        self.iou.reset()
        self.batch_loss.reset()
        # print("Train Epoch Ended")

    def on_validation_epoch_end(self):
        self.acc_1.reset()
        self.acc_3.reset()
        self.iou.reset()
        self.batch_loss.reset()
        # print("Valid Epoch Ended")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return optimizer
