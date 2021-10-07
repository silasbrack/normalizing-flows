from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize
import pytorch_lightning as pl
from typing import Optional


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 64, train_size: int = 55000, val_size: int = 5000):
        super().__init__()

        self.data_dir = "data/"
        self.transform = Compose([ToTensor(),
                                  Normalize((0.1307,), (0.3081,))])
        self.batch_size = batch_size
        self.train_size = train_size
        self.val_size = val_size

    def prepare_data(self):
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):

        # Assign Train/val split(s) for use in Dataloaders
        if stage in (None, 'fit'):
            mnist_full = MNIST(
                self.data_dir,
                train=True,
                download=False,
                transform=self.transform
            )
            self.mnist_train, self.mnist_val = random_split(mnist_full, [self.train_size, self.val_size])
            self.dims = self.mnist_train[0][0].shape

        # Assign Test split(s) for use in Dataloaders
        if stage in (None, 'test'):
            self.mnist_test = MNIST(
                self.data_dir,
                train=False,
                download=False,
                transform=self.transform
            )
            self.dims = getattr(self, 'dims', self.mnist_test[0][0].shape)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.mnist_train, batch_size=self.batch_size)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.mnist_val, batch_size=self.batch_size)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.mnist_test, batch_size=self.batch_size)
