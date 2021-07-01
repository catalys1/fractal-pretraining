from typing import Optional, Tuple

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR100
from torchvision.transforms import transforms
from torchvision.transforms.functional import InterpolationMode

from .utils import to_tensor


__all__ = ['Cifar100DataModule']

class Cifar100DataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 64,
        num_workers: int = 4,
        pin_memory: bool = True,
        size: int = 32,
        augment: bool = True,
        normalize: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.augment = augment
        self.normalize = normalize

        self.dims = (3, size, size)
        self.num_class = 100

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def prepare_data(self):
        CIFAR100(self.data_dir, download=True)

    def setup(self, stage):
        pass

    def train_dataloader(self):
        if self.data_train is None:
            tform = transforms.Compose([
                transforms.Resize(self.dims[1:], interpolation=InterpolationMode.BILINEAR),
                transforms.Pad(self.dims[1] // 8, padding_mode='reflect'),
                transforms.RandomAffine((-10, 10), (0, 1/8), (1, 1.2), interpolation=InterpolationMode.BILINEAR),
                transforms.CenterCrop(self.dims[1:]),
                transforms.RandomHorizontalFlip(0.5),
                to_tensor(self.normalize),
            ])
            self.data_train = CIFAR100(
                root = self.data_dir,
                train = True, 
                transform = tform if self.augment else to_tensor(self.normalize)
            )

        return DataLoader(
            dataset = self.data_train,
            batch_size = self.batch_size,
            num_workers = self.num_workers,
            pin_memory = self.pin_memory,
            shuffle = True,
            drop_last = True
        )

    def val_dataloader(self):
        if self.data_val is None:
            tform = transforms.Compose([
                transforms.Resize(self.dims[1:], interpolation=InterpolationMode.BILINEAR),
                to_tensor(self.normalize),
            ])
            self.data_val = CIFAR100(
                root = self.data_dir,
                train = False, 
                transform = tform
            )

        return DataLoader(
            dataset = self.data_val,
            batch_size = self.batch_size,
            num_workers = self.num_workers,
            pin_memory = self.pin_memory,
            shuffle = False,
            drop_last = False
        )