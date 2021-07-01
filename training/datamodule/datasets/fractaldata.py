from functools import partial
import pickle
from typing import Callable, Optional, Tuple, Union
import warnings

from cv2 import GaussianBlur
import numpy as np
import torch
import torchvision
# from torchvision.transforms.functional import InterpolationMode

from fractals import diamondsquare, ifs


class IFSGenerator(object):
    def __init__(
        self,
        size: int = 224,
        jitter_params: Union[bool, str] = True,
        flips: bool = True,
        scale: Optional[Tuple[float, float]] = (0.5, 2.0),
        translate: Optional[float] = 0.2,
        sigma: Optional[Tuple[float, float]] = (0.5, 1.0),
        blur_p: Optional[float] = 0.5,
        color = True,
        background = True,
        niter = 100000,
        patch = True,
    ):
        self.size = size
        self.jitter_params = jitter_params
        self.flips = flips
        self.scale = scale
        self.translate = translate
        self.sigma = sigma
        self.blur_p = blur_p
        self.color = color
        self.background = background
        self.niter = niter
        self.patch = patch

        if isinstance(self.jitter_params, str) and self.jitter_params.startswith('fractaldb'):
            k = int(self.jitter_params.split('-')[1]) / 10
            choices = np.linspace(1-2*k, 1+2*k, 5, endpoint=True)
            self.jitter_fnc = partial(self.fractaldb_jitter, choices=choices)
        else:
            self.jitter_fnc = self.basic_jitter

    def fractaldb_jitter(self, sys, rng, choices=(.8,.9,1,1.1,1.2)):
        n = len(sys)
        y, x = np.divmod(rng.integers(0, 6, (n,)), 3)
        sys[range(n), y, x] *= rng.choice(choices)
        return sys

    def basic_jitter(self, sys, rng, prange=(0.8, 1.1)):
        # tweak system parameters--randomly choose one transform and scale it
        n = len(sys)
        sys[rng.integers(0, n)] *= rng.uniform(*prange)
        return sys

    def generate(self, sys, ps=None):
        rng = np.random.default_rng()

        attempts = 4 if self.jitter_params else 0
        for i in range(attempts):
            # jitter system parameters
            sysc = sys.copy()
            sysc = self.jitter_fnc(sysc, rng)
            svd = np.linalg.svd(sysc[:,:,:2], compute_uv=False)
            if svd.max() > 1: continue
            coords = ifs.iterate(sysc, self.niter, ps)
            region = np.concatenate(ifs.minmax(coords))
            break
            # occasionally the modified parameters cause the system to explode
            # if np.all(np.isfinite(region)): break
        else:
            # fall back on not jittering the parameters
            coords = ifs.iterate(sys, self.niter, ps)
            region = np.concatenate(ifs.minmax(coords))

        # transform rendering window (scale and translate)
        if self.translate or self.scale:
            extent = (region[2:] - region[:2])
            center = (region[2:] + region[:2]) / 2
            if self.translate:
                center += extent * rng.uniform(-self.translate, self.translate, (2,))
            if self.scale:
                extent *= rng.uniform(*self.scale, (2,)) / 2
            region[:2] = center - extent
            region[2:] = center + extent
            # coords = coords[(coords[:,0] > region[0]) & (coords[:,0] < region[2]) & 
            #                 (coords[:,1] > region[1]) & (coords[:,1] < region[3])]

        img = ifs.render(coords, self.size, binary=not self.color, region=region, patch=self.patch)

        # random flips/rotations
        if rng.random() > 0.5:
            img = img.transpose(1, 0)
        if rng.random() > 0.5:
            img = img[::-1]
        if rng.random() > 0.5:
            img = img[:, ::-1]
        img = np.ascontiguousarray(img)

        idx = img == 0

        # colorize
        if self.color:
            img = ifs.colorize(img)
        else:
            img = (img * 127).astype(np.uint8)[..., None].repeat(3, axis=2)

        # add random background
        if self.background:
            bg = diamondsquare.colorized_ds(self.size)
            img[idx] = bg[idx]

        # randomly apply gaussian blur
        if self.blur_p and rng.random() > 0.5:
            sigma = rng.uniform(*self.sigma)
            img = GaussianBlur(img, (3, 3), sigma, dst=img)

        return img

    def __call__(self, ws, ps=None):
        return self.generate(ws, ps)


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
        ws = self.params[label]['ws']
        img = self.generator(ws)
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
