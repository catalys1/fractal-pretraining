from typing import Callable, Optional, Tuple

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms

from .datasets import fractaldata


class FractalClassDataModule(LightningDataModule):
    """
    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
        self,
        data_dir: str = "data/",
        batch_size: int = 64,
        num_workers: int = 4,
        pin_memory: bool = True,
        size: int = 256,
        data_file: str = None,
        num_class: int = 1000,
        per_class: int = 100,
        generator: Optional[Callable] = None,
        normalize: Optional[str] = None,
        queue_size: int = 0,
        **kwargs,
    ):
        super().__init__()

        self.data_dir = data_dir
        self.data_file = data_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.num_class = num_class
        self.per_class = per_class
        self.generator = generator
        self.normalize = normalize
        self.queue_size = queue_size

        # self.dims is returned when you call datamodule.size()
        self.dims = (3, size, size)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: self.data_train, self.data_val, self.data_test."""
        if self.data_file is None:
            self.data_file = self.data_dir + 'ifs_constrained-search-15k.pkl'
        else:
            self.data_file = self.data_dir + self.data_file
        self.data_train = fractaldata.FractalClassDataset(
            self.data_file, self.num_class, self.per_class, self.generator, self.queue_size)
        self.data_val = self.data_train
        self.data_test = None

    def train_dataloader(self):
        if self.data_train is None:
            self.setup()
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        if self.data_val is None:
            self.setup()
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            drop_last=False,
        )
