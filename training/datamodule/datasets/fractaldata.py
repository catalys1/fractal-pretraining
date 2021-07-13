from functools import partial
import pickle
from typing import Callable, Optional, Tuple, Union
import warnings

from cv2 import GaussianBlur
import numpy as np
import torch
import torchvision

from fractals import diamondsquare, ifs
from .generator import IFSGenerator, MultiGenerator


class FractalClassDataset(object):
    def __init__(
        self,
        param_file: str,
        num_class: int = 1000,
        per_class: int = 100,
        generator: Optional[Callable] = None,
        queue_size: int = 0,
    ):
        self.params = pickle.load(open(param_file, 'rb'))['params'][:num_class]
        self.num_class = num_class
        self.per_class = per_class

        self.generator = generator or IFSGenerator()

        self.queue_size = queue_size
        self.queue = None
        if queue_size > 0:
            self.queue = []
            T = torchvision.transforms
            self.queue_tform = torchvision.transforms.Compose([
                T.RandomAffine(10, (0.1, 0.1), (0.7, 0.95), interpolation=T.functional.InterpolationMode.BILINEAR),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip()
            ])

    def render_img(self, idx):
        label = idx % self.num_class
        params = self.params[label]['ws']
        img = self.generator(params)
        return img

    def get_img(self, idx):
        img = self.render_img(idx)
        img = torch.from_numpy(img).float().mul_(1/255.).permute(2,0,1)
        label = idx % self.num_class
        return img, label

    def __len__(self):
        return self.num_class * self.per_class

    def __getitem__(self, idx):
        if self.queue is None or len(self.queue) < self.queue_size:
            img, label = self.get_img(idx)
            if self.queue is not None:
                self.queue.append((img, label))
        elif np.random.default_rng().random() < 0.5:
            img, label = self.get_img(idx) 
            self.queue[idx % self.queue_size] = (img, label)
        else:
            img, label = self.queue[idx % self.queue_size]
            img = self.queue_tform(img)

        return img, label


class MultiFractalDataset(object):
    def __init__(
        self,
        param_file: str,
        num_class: int = 10000,
        dataset_size: int = 1000000,
        generator: Optional[Callable] = None,
        period: int = 2,
    ):
        self.params = pickle.load(open(param_file, 'rb'))['params'][:num_class]
        self.num_class = num_class
        self.dataset_size = dataset_size

        self.generator = generator or MultiGenerator()
        # start with an image in the cache
        k = np.random.default_rng().integers(0, num_class)
        self.generator.add_sample(self.params[k]['ws'], label=k)

        self.steps = 0
        self.period = period

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        self.steps = (self.steps + 1) % self.period
        sample = self.steps == 0
        label = idx % self.num_class
        params = self.params[label]['ws']
        img, labels = self.generator(params, label=label, new_sample=sample)
        img = torch.from_numpy(img).float().mul_(1/255.).permute(2,0,1)

        return img, labels
        
