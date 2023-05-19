import pytorch_lightning as pl
import torchvision.datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
import os


class MnistRecData(pl.LightningDataModule):

    def __init__(self,
                 batch_size=32,
                 num_workers=8,
                 validation_ratio=0.2,
                 data_path='./data/MNIST'):
        super(MnistRecData, self).__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.validation_ratio = validation_ratio
        self.data_path = data_path

    def prepare_data(self) -> None:
        # download
        torchvision.datasets.MNIST(self.data_path, train=True, download=True)
        torchvision.datasets.MNIST(self.data_path, train=False, download=True)

    def setup(self, stage=None):
        print(1111)
        # transform
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        mnist_train = torchvision.datasets.MNIST(self.data_path, train=True, download=False, transform=transform)
        mnist_test = torchvision.datasets.MNIST(self.data_path, train=False, download=False, transform=transform)
        # 划分验证集和训练集
        train_size = int((1 - self.validation_ratio) * len(mnist_train))
        validation_size = len(mnist_train) - train_size

        mnist_train, mnist_val = torch.utils.data.random_split(mnist_train, [train_size, validation_size])

        # assign to use in dataloaders
        self.train_dataset = mnist_train
        self.val_dataset = mnist_val
        self.test_dataset = mnist_test

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)


if __name__ == '__main__':
    data_module = MnistRecData()
    data_module.prepare_data()
    data_module.setup('train')
    data = data_module.train_dataloader()
