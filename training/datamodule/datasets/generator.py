from functools import partial
from typing import Callable, Optional, Tuple, Union

from cv2 import GaussianBlur, resize, INTER_LINEAR
import numpy as np

from fractals import diamondsquare, ifs


class _GeneratorBase(object):
    def __init__(
        self,
        size: int = 224,
        jitter_params: Union[bool, str] = True,
        flips: bool = True,
        sigma: Optional[Tuple[float, float]] = (0.5, 1.0),
        blur_p: Optional[float] = 0.5,
        niter = 100000,
        patch = True,
    ):
        self.size = size
        self.jitter_params = jitter_params
        self.flips = flips
        self.sigma = sigma
        self.blur_p = blur_p
        self.niter = niter
        self.patch = patch

        self.rng = np.random.default_rng()

        self.cache = {'fg': [], 'bg': []}

        if isinstance(self.jitter_params, str) and self.jitter_params.startswith('fractaldb'):
            k = int(self.jitter_params.split('-')[1]) / 10
            choices = np.linspace(1-2*k, 1+2*k, 5, endpoint=True)
            self.jitter_fnc = partial(self._fractaldb_jitter, choices=choices)
        else:
            self.jitter_fnc = self._basic_jitter

    def _fractaldb_jitter(self, sys, choices=(.8,.9,1,1.1,1.2)):
        n = len(sys)
        y, x = np.divmod(self.rng.integers(0, 6, (n,)), 3)
        sys[range(n), y, x] *= self.rng.choice(choices)
        return sys

    def _basic_jitter(self, sys, prange=(0.8, 1.1)):
        # tweak system parameters--randomly choose one transform and scale it
        n = len(sys)
        sys[self.rng.integers(0, n)] *= self.rng.uniform(*prange)
        return sys

    def jitter(self, sys):
        attempts = 4 if self.jitter_params else 0
        for i in range(attempts):
            # jitter system parameters
            sysc = sys.copy()
            sysc = self.jitter_fnc(sysc)
            # occasionally the modified parameters cause the system to explode
            svd = np.linalg.svd(sysc[:,:,:2], compute_uv=False)
            if svd.max() > 1: continue
            break
        else:
            # fall back on not jittering the parameters
            sysc = sys
        return sysc
    
    def _iterate(self, sys):
        rng = self.rng

        coords = ifs.iterate(sys, self.niter)
        region = np.concatenate(ifs.minmax(coords))

        return coords, region

    def render(self, sys):
        raise NotImplementedError()

    def random_flips(self, img):
        # random flips/rotations
        if self.rng.random() > 0.5:
            img = img.transpose(1, 0)
        if self.rng.random() > 0.5:
            img = img[::-1]
        if self.rng.random() > 0.5:
            img = img[:, ::-1]
        img = np.ascontiguousarray(img)
        return img

    def to_color(self, img):
        return ifs.colorize(img)

    def to_gray(self, img):
        return (img * 127).astype(np.uint8)[..., None].repeat(3, axis=2)

    def render_background(self):
        bg = diamondsquare.colorized_ds(self.size)
        return bg

    def composite(self, foreground, base, idx=None):
        return ifs.composite(foreground, base)

    def random_blur(self, img):
        sigma = self.rng.uniform(*self.sigma)
        img = GaussianBlur(img, (3, 3), sigma, dst=img)
        return img

    def generate(self, sys):
        raise NotImplementedError()

    def __call__(self, sys, *args, **kwargs):
        return self.generate(sys, *args, **kwargs)


class IFSGenerator(_GeneratorBase):
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

        self.rng = np.random.default_rng()

        self.cache = {'fg': None, 'bg': None}

        if isinstance(self.jitter_params, str) and self.jitter_params.startswith('fractaldb'):
            k = int(self.jitter_params.split('-')[1]) / 10
            choices = np.linspace(1-2*k, 1+2*k, 5, endpoint=True)
            self.jitter_fnc = partial(self._fractaldb_jitter, choices=choices)
        else:
            self.jitter_fnc = self._basic_jitter

    def render(self, sys):
        rng = self.rng
        coords, region = self._iterate(sys)

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

        img = ifs.render(coords, self.size, binary=not self.color, region=region, patch=self.patch)
        return img

    def generate(self, sys):
        rng = self.rng

        sysc = self.jitter(sys)
        img = self.render(sysc)
        self.cache['fg'] = img

        # random flips
        if self.flips:
            img = self.random_flips(img)

        # colorize
        if self.color:
            img = self.to_color(img)
        else:
            img = self.to_gray(img)

        # add random background
        if self.background:
            bg = self.render_background()
            self.cache['bg'] = bg.copy()
            img = self.composite(img, bg)

        # randomly apply gaussian blur
        if self.blur_p and rng.random() > 0.5:
            img = self.random_blur(img)

        return img


class MultiGenerator(_GeneratorBase):
    def __init__(
        self,
        size: int = 224,
        cache_size: int = 512,
        n_objects: Tuple[int, int] = (1, 5),
        size_range: Tuple[float, float] = (0.15, 0.6),
        jitter_params: Union[bool, str] = True,
        flips: bool = True,
        sigma: Optional[Tuple[float, float]] = (0.5, 1.0),
        blur_p: Optional[float] = 0.5,
        niter = 100000,
        patch = True,
    ):
        self.size = size
        self.n_objects = n_objects
        self.size_range = size_range
        self.jitter_params = jitter_params
        self.flips = flips
        self.sigma = sigma
        self.blur_p = blur_p
        self.niter = niter
        self.patch = patch

        self.rng = np.random.default_rng()

        self.cache_size = cache_size
        self.cache = {'fg': [], 'bg': [], 'label': []}
        self.idx = 0

        if isinstance(self.jitter_params, str) and self.jitter_params.startswith('fractaldb'):
            k = int(self.jitter_params.split('-')[1]) / 10
            choices = np.linspace(1-2*k, 1+2*k, 5, endpoint=True)
            self.jitter_fnc = partial(self._fractaldb_jitter, choices=choices)
        else:
            self.jitter_fnc = self._basic_jitter

    def __len__(self):
        return len(self.cache['fg'])

    def _update_cache(self, fg, bg, label):
        if len(self) < self.cache_size:
            self.cache['fg'].append(fg)
            self.cache['bg'].append(bg)
            self.cache['label'].append(label)
        else:
            self.cache['fg'][self.idx] = fg
            self.cache['bg'][self.idx] = bg
            self.cache['label'][self.idx] = label
        self.idx = (self.idx + 1) % self.cache_size

    def render(self, sys):
        rng = self.rng
        coords, region = self._iterate(sys)
        # render the fractal at half resolution--it will be resized during generation phase
        img = ifs.render(coords, self.size // 2, binary=False, region=region, patch=self.patch)
        return img

    def add_sample(self, sys, label=-1):
        sysc = self.jitter(sys)
        frac = self.render(sysc)
        bg = self.render_background()
        self._update_cache(frac, bg, label)

    def generate(self, sys, label=-1, new_sample=True):
        rng = self.rng

        if new_sample:
            self.add_sample(sys, label)

        idx = rng.integers(0, len(self))
        img = self.cache['bg'][idx].copy()
        labels = []

        n = rng.integers(*self.n_objects, endpoint=True)
        n = min(n, len(self))
        for i in range(n):
            idx = rng.integers(0, len(self))
            labels.append(self.cache['label'][idx])
            fg = self.cache['fg'][idx]
            # random flips
            if self.flips:
                fg = self.random_flips(fg)
            fg = self.to_color(fg)
            # random size
            f = rng.uniform(*self.size_range)
            s = int(f * self.size)
            fg = resize(fg, (s, s), interpolation=INTER_LINEAR)
            # random location
            x, y = rng.integers(-(s//2), self.size-(s//2), 2)
            x1 = 0 if x >= 0 else -x
            x2 = s if x < self.size - s else self.size - x
            y1 = 0 if y >= 0 else -y
            y2 = s if y < self.size - s else self.size - y
            fg = fg[y1:y2, x1:x2]
            # add object to image
            y = max(y, 0)
            x = max(x, 0)
            self.composite(fg, img[y:y+fg.shape[0], x:x+fg.shape[1]])

        # randomly apply gaussian blur
        if self.blur_p and rng.random() > 0.5:
            img = self.random_blur(img)

        return img, labels