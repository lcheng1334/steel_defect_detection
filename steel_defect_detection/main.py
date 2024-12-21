

from data_preprocessing import SteelDataset
from model import UNet
from train_and_eval import TrainAndEvalLightning

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

torch.set_float32_matmul_precision('medium')




def main():
    transform = transforms.Compose([
        transforms.Resize((256, 200)),
    ])
    train_set = SteelDataset(train=True, transform=transform)
    test_set = SteelDataset(train=False, transform=transform)
    trainset, valset = random_split(train_set, [0.8, 0.2])

    batch_size = 16

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=12)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=12)
    # test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=8)

    model = UNet(n_channels=1, n_classes=16)

    train_module = TrainAndEvalLightning(model=model, lr=0.01)
    logger = TensorBoardLogger(save_dir='./logs', name='medium_precision')
    trainer = Trainer(max_epochs=3, logger=logger)
    trainer.fit(train_module, train_loader, val_loader)


if __name__ == '__main__':
    main()