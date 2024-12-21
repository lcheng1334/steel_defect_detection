# -*- coding:utf-8 -*-
# Time: 2024/12/21
# Author: lcheng1334
# File: lightning.py
import pytorch_lightning as pl
import torch
from torch import nn
from torchmetrics import Accuracy, Precision, Dice


class LightningModel(pl.LightningModule):
    def __init__(self, model, lr):
        super().__init__()
        self.model = model
        self.lr = lr
        self.loss_fn = nn.CrossEntropyLoss()

        # 训练指标
        self.train_accuracy = Accuracy(task="binary")
        self.train_precision = Precision(task="binary")
        self.train_dice = Dice(num_classes=2, average="macro")

        # 验证指标
        self.val_accuracy = Accuracy(task="binary")
        self.val_precision = Precision(task="binary")
        self.val_dice = Dice(num_classes=2, average="macro")

        # 测试指标
        self.test_accuracy = Accuracy(task="binary")
        self.test_precision = Precision(task="binary")
        self.test_dice = Dice(num_classes=2, average="macro")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)

        # 计算损失
        loss = self.loss_fn(y_hat, y.long())
        y_pred = torch.argmax(y_hat, dim=1)
        # 更新训练指标
        self.train_accuracy.update(y_pred, y.int())
        self.train_precision.update(y_pred, y.int())
        self.train_dice.update(y_pred, y.int())

        # 记录当前批次的损失
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)

        # 计算损失
        loss = self.loss_fn(y_hat,  y.long())
        y_pred = torch.argmax(y_hat, dim=1)
        # 更新验证指标
        self.val_accuracy.update(y_pred, y.int())
        self.val_precision.update(y_pred, y.int())
        self.val_dice.update(y_pred, y.int())

        # 记录当前批次的损失
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)

        # 计算损失
        loss = self.loss_fn(y_hat, y.long())
        y_pred = torch.argmax(y_hat, dim=1)

        # 更新测试指标
        self.test_accuracy.update(y_pred, y.int())
        self.test_precision.update(y_pred, y.int())
        self.test_dice.update(y_pred, y.int())

        # 记录当前批次的损失
        self.log('test_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def on_train_epoch_end(self):
        # 记录整个训练 epoch 的指标
        self.log('train_accuracy', self.train_accuracy.compute(), prog_bar=True)
        self.log('train_precision', self.train_precision.compute(), prog_bar=True)
        self.log('train_dice', self.train_dice.compute(), prog_bar=True)

        # 重置状态
        self.train_accuracy.reset()
        self.train_precision.reset()
        self.train_dice.reset()

    def on_validation_epoch_end(self):
        # 记录整个验证 epoch 的指标
        self.log('val_accuracy', self.val_accuracy.compute(), prog_bar=True)
        self.log('val_precision', self.val_precision.compute(), prog_bar=True)
        self.log('val_dice', self.val_dice.compute(), prog_bar=True)

        # 重置状态
        self.val_accuracy.reset()
        self.val_precision.reset()
        self.val_dice.reset()

    def on_test_epoch_end(self):
        # 记录整个测试 epoch 的指标
        self.log('test_accuracy', self.test_accuracy.compute(), prog_bar=True)
        self.log('test_precision', self.test_precision.compute(), prog_bar=True)
        self.log('test_dice', self.test_dice.compute(), prog_bar=True)

        # 重置状态
        self.test_accuracy.reset()
        self.test_precision.reset()
        self.test_dice.reset()
