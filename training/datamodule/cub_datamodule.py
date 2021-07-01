from typing import Optional, Tuple

from fgvcdata import CUB
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms

from .utils import to_tensor


__all__ = ['CubDataModule']


class CubDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 64,
        num_workers: int = 4,
        pin_memory: bool = True,
        size: int = 224,
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
        self.num_class = 200

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def prepare_data(self):
        pass

    def setup(self, stage):
        pass

    def train_dataloader(self):
        if self.data_train is None:
            tform = transforms.Compose([
                transforms.RandomResizedCrop(self.dims[1:], scale=(0.2, 1.0)),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ColorJitter(0.25, 0.25, 0.25),
                to_tensor(self.normalize),
            ])
            self.data_train = CUB(
                root = f'{self.data_dir}/train',
                transform = tform if self.augment else to_tens
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
                transforms.Resize([int(round(8 * x / 7)) for x in self.dims[1:]]),
                transforms.CenterCrop(self.dims[1:]),
                to_tensor(self.normalize),
            ])
            self.data_val = CUB(
                root = f'{self.data_dir}/val',
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