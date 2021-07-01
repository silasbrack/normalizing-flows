import pytest
import torch
from src.data import MNISTDataModule


@pytest.fixture
def data():
    data = MNISTDataModule()
    data.prepare_data()
    data.setup()
    return data


def test_constructor(data):
    assert type(data) == MNISTDataModule


def test_setup(data):
    assert len(data.mnist_train) == 55000
    assert len(data.mnist_val) == 5000
    assert len(data.mnist_test) == 10000


def test_train_dataloader(data):
    data_loader = data.train_dataloader()
    batch = next(iter(data_loader))
    assert batch[0].shape == torch.Size([64, 1, 28, 28])
    assert batch[1].shape == torch.Size([64])
    assert len(data_loader.dataset) == 55000


def test_val_dataloader(data):
    data_loader = data.val_dataloader()
    batch = next(iter(data_loader))
    assert batch[0].shape == torch.Size([64, 1, 28, 28])
    assert batch[1].shape == torch.Size([64])
    assert len(data_loader.dataset) == 5000


def test_test_dataloader(data):
    data_loader = data.test_dataloader()
    batch = next(iter(data_loader))
    assert batch[0].shape == torch.Size([64, 1, 28, 28])
    assert batch[1].shape == torch.Size([64])
    assert len(data_loader.dataset) == 10000
