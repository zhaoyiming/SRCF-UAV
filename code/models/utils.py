import os
import tempfile
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from numpy import random


class GridMask(nn.Module):
    def __init__(self, ratio=0.5, prob=0.7):
        super(GridMask, self).__init__()
        self.ratio = ratio
        self.prob = prob

    def forward(self, x):
        if np.random.rand() > self.prob or not self.training:
            return x

        n, c, h, w = x.size()
        x = x.view(-1, h, w)
        hh = int(1.5 * h)
        ww = int(1.5 * w)

        d = np.random.randint(2, h)
        l = min(max(int(d * self.ratio + 0.5), 1), d - 1)
        mask = np.ones((hh, ww), np.uint8)
        st_h = np.random.randint(d)
        st_w = np.random.randint(d)

        for i in range(hh // d):
            s = d*i + st_h
            t = min(s + l, hh)
            mask[s:t, :] = 0

        for i in range(ww // d):
            s = d*i + st_w
            t = min(s + l, ww)
            mask[:, s:t] = 0

        mask = mask[(hh-h)//2:(hh-h)//2+h, (ww-w)//2:(ww-w)//2+w]
        mask = torch.tensor(mask, dtype=x.dtype, device=x.device)
        mask = 1 - mask
        mask = mask.expand_as(x)
        x = x * mask 

        return x.view(n, c, h, w)


def rotation_3d_in_axis(points, angles):
    assert points.shape[-1] == 3
    assert angles.shape[-1] == 1
    angles = angles[..., 0]

    n_points = points.shape[-2]
    input_dims = angles.shape

    if len(input_dims) > 1:
        points = points.reshape(-1, n_points, 3)
        angles = angles.reshape(-1)

    rot_sin = torch.sin(angles)
    rot_cos = torch.cos(angles)
    ones = torch.ones_like(rot_cos)
    zeros = torch.zeros_like(rot_cos)

    if VERSION.name == 'v0.17.1':
        rot_mat_T = torch.stack([
            rot_cos, -rot_sin, zeros,
            rot_sin, rot_cos, zeros,
            zeros, zeros, ones,
        ]).transpose(0, 1).reshape(-1, 3, 3)
    else:
        rot_mat_T = torch.stack([
            rot_cos, rot_sin, zeros,
            -rot_sin, rot_cos, zeros,
            zeros, zeros, ones,
        ]).transpose(0, 1).reshape(-1, 3, 3)

    points = torch.bmm(points, rot_mat_T)

    if len(input_dims) > 1:
        points = points.reshape(*input_dims, n_points, 3)
    
    return points


def inverse_sigmoid(x, eps=1e-5):
    """Inverse function of sigmoid.
    Args:
        x (Tensor): The tensor to do the
            inverse.
        eps (float): EPS avoid numerical
            overflow. Defaults 1e-5.
    Returns:
        Tensor: The x has passed the inverse
            function of sigmoid, has same
            shape with input.
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


def pad_multiple(inputs, img_metas, size_divisor=32):
    _, _, img_h, img_w = inputs.shape

    pad_h = 0 if img_h % size_divisor == 0 else size_divisor - (img_h % size_divisor)
    pad_w = 0 if img_w % size_divisor == 0 else size_divisor - (img_w % size_divisor)

    B = len(img_metas)
    N = len(img_metas[0]['ori_shape'])

    for b in range(B):
        img_metas[b]['img_shape'] = [(img_h + pad_h, img_w + pad_w, 3) for _ in range(N)]
        img_metas[b]['pad_shape'] = [(img_h + pad_h, img_w + pad_w, 3) for _ in range(N)]

    if pad_h == 0 and pad_w == 0:
        return inputs
    else:
        return F.pad(inputs, [0, pad_w, 0, pad_h], value=0)


def rgb_to_hsv(image: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    r"""Convert an image from RGB to HSV.

    .. image:: _static/img/rgb_to_hsv.png

    The image data is assumed to be in the range of (0, 1).

    Args:
        image: RGB Image to be converted to HSV with shape of :math:`(*, 3, H, W)`.
        eps: scalar to enforce numarical stability.

    Returns:
        HSV version of the image with shape of :math:`(*, 3, H, W)`.
        The H channel values are in the range 0..2pi. S and V are in the range 0..1.

    .. note::
       See a working example `here <https://kornia-tutorials.readthedocs.io/en/latest/
       color_conversions.html>`__.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = rgb_to_hsv(input)  # 2x3x4x5
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    image = image / 255.0

    max_rgb, argmax_rgb = image.max(-3)
    min_rgb, argmin_rgb = image.min(-3)
    deltac = max_rgb - min_rgb

    v = max_rgb
    s = deltac / (max_rgb + eps)

    deltac = torch.where(deltac == 0, torch.ones_like(deltac), deltac)
    rc, gc, bc = torch.unbind((max_rgb.unsqueeze(-3) - image), dim=-3)

    h1 = bc - gc
    h2 = (rc - bc) + 2.0 * deltac
    h3 = (gc - rc) + 4.0 * deltac

    h = torch.stack((h1, h2, h3), dim=-3) / deltac.unsqueeze(-3)
    h = torch.gather(h, dim=-3, index=argmax_rgb.unsqueeze(-3)).squeeze(-3)
    h = (h / 6.0) % 1.0

    h = h * 360.0
    v = v * 255.0

    return torch.stack((h, s, v), dim=-3)


def hsv_to_rgb(image: torch.Tensor) -> torch.Tensor:
    r"""Convert an image from HSV to RGB.

    The H channel values are assumed to be in the range 0..2pi. S and V are in the range 0..1.

    Args:
        image: HSV Image to be converted to HSV with shape of :math:`(*, 3, H, W)`.

    Returns:
        RGB version of the image with shape of :math:`(*, 3, H, W)`.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = hsv_to_rgb(input)  # 2x3x4x5
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    h: torch.Tensor = image[..., 0, :, :] / 360.0
    s: torch.Tensor = image[..., 1, :, :]
    v: torch.Tensor = image[..., 2, :, :] / 255.0

    hi: torch.Tensor = torch.floor(h * 6) % 6
    f: torch.Tensor = ((h * 6) % 6) - hi
    one: torch.Tensor = torch.tensor(1.0, device=image.device, dtype=image.dtype)
    p: torch.Tensor = v * (one - s)
    q: torch.Tensor = v * (one - f * s)
    t: torch.Tensor = v * (one - (one - f) * s)

    hi = hi.long()
    indices: torch.Tensor = torch.stack([hi, hi + 6, hi + 12], dim=-3)
    out = torch.stack((v, q, p, p, t, v, t, v, v, q, p, p, p, p, t, v, v, q), dim=-3)
    out = torch.gather(out, -3, indices)
    out = out * 255.0

    return out


class GpuPhotoMetricDistortion:
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.
    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    8. randomly swap channels
    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18,
                 seed=None):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta
        self.seed = seed
        # Create a local random state generator for reproducibility
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        else:
            self.rng = None
        # Cache for sequence-consistent augmentation: only store random seed per id
        # Key: sample_index (sequence_token hash), Value: random seed offset
        self._aug_seed_cache = {}
        # Track active id set: when id set changes, generate new augmentation for new ids
        # Key: id, Value: random seed offset (only updated when id set changes)
        self._active_ids = set()  # Track current active id set

    def __call__(self, imgs, sample_indices=None):
        """Call function to perform photometric distortion on images.
        Args:
            imgs (Tensor): Input images with shape (B, C, H, W).
            sample_indices (list, optional): List of indices for each sample in batch.
                If provided, samples with same index will get same augmentation.
                If None (default), uses batch index (0, 1, 2, ...) - original behavior.
        Returns:
            Tensor: Distorted images with same shape.
        """
        batch_size = imgs.shape[0]
        
        # Use provided sample_indices or default to batch indices (original behavior)
        if sample_indices is None:
            sample_indices = list(range(batch_size))
        else:
            assert len(sample_indices) == batch_size, f"sample_indices length {len(sample_indices)} != batch_size {batch_size}"
        
        # Check if id set has changed: only generate new augmentation for new ids
        current_ids = set(sample_indices)
        ids_changed = (current_ids != self._active_ids)
        # new_ids = set()
        
        if ids_changed:
            # Id set changed: generate new augmentation seed for new ids only
            new_ids = current_ids - self._active_ids
            for uid in new_ids:
                # Generate a new random seed offset for new id
                if self.seed is not None:
                    if self.rng is not None:
                        seed_offset = self.rng.randint(0, 2**31)
                    else:
                        seed_offset = np.random.randint(0, 2**31)
                    self._aug_seed_cache[uid] = seed_offset
            # Update active id set
            self._active_ids = current_ids.copy()
        
        # Print augmentation info for this batch
        # if self.seed is not None:
        #     unique_ids = sorted(list(set(sample_indices)))
        #     aug_info = {uid: self._aug_seed_cache.get(uid, None) for uid in unique_ids}
        #     if ids_changed:
        #         new_ids_list = sorted(list(new_ids))
        #         print(f"[Augmentation] Batch ids: {unique_ids}, New ids: {new_ids_list}, Aug seeds: {aug_info}")
        #     else:
        #         print(f"[Augmentation] Batch ids: {unique_ids}, (no change), Aug seeds: {aug_info}")
        
        # Create RNG functions: use cached seed for each id
        unique_ids = list(set(sample_indices))
        if self.seed is not None:
            # Pre-generate random number sequences for each unique id
            id_randint_seqs = {}  # id -> list of randint values
            id_uniform_seqs = {}  # id -> list of uniform values
            id_permutation_seqs = {}  # id -> list of permutation arrays
            
            for uid in unique_ids:
                seed_offset = self._aug_seed_cache[uid]
                rng = np.random.RandomState(self.seed + seed_offset)
                
                # Pre-generate sequences (enough for one batch)
                id_randint_seqs[uid] = [rng.randint(0, 2) for _ in range(20)]
                id_uniform_seqs[uid] = [rng.uniform(0, 1) for _ in range(10)]
                id_permutation_seqs[uid] = [rng.permutation(3) for _ in range(5)]
            
            # Track position in sequence for each id (reset each batch)
            id_randint_pos = {uid: 0 for uid in unique_ids}
            id_uniform_pos = {uid: 0 for uid in unique_ids}
            id_permutation_pos = {uid: 0 for uid in unique_ids}
            
            def randint(idx, *args, **kwargs):
                uid = sample_indices[idx]
                pos = id_randint_pos[uid]
                max_val = args[0] if args else 2
                result = id_randint_seqs[uid][pos] % max_val
                id_randint_pos[uid] = (pos + 1) % len(id_randint_seqs[uid])
                return result
            
            def uniform(idx, low, high):
                uid = sample_indices[idx]
                pos = id_uniform_pos[uid]
                result = low + id_uniform_seqs[uid][pos] * (high - low)
                id_uniform_pos[uid] = (pos + 1) % len(id_uniform_seqs[uid])
                return result
            
            def permutation(idx, n):
                uid = sample_indices[idx]
                pos = id_permutation_pos[uid]
                result = id_permutation_seqs[uid][pos].copy()
                id_permutation_pos[uid] = (pos + 1) % len(id_permutation_seqs[uid])
                return result
        else:
            # Use global random state (not reproducible)
            randint = lambda idx, *args, **kwargs: random.randint(*args, **kwargs)
            uniform = lambda idx, *args, **kwargs: random.uniform(*args, **kwargs)
            permutation = lambda idx, *args, **kwargs: random.permutation(*args, **kwargs)

        imgs = imgs[:, [2, 1, 0], :, :]  # BGR to RGB

        # Generate contrast modes for each sample
        contrast_modes = []
        for idx in range(batch_size):
            # mode == 0 --> do random contrast first
            # mode == 1 --> do random contrast last
            contrast_modes.append(randint(idx, 2))

        # Apply brightness and contrast mode 0
        for idx in range(batch_size):
            # random brightness
            if randint(idx, 2):
                delta = uniform(idx, -self.brightness_delta, self.brightness_delta)
                imgs[idx] += delta

            if contrast_modes[idx] == 0:
                if randint(idx, 2):
                    alpha = uniform(idx, self.contrast_lower, self.contrast_upper)
                    imgs[idx] *= alpha

        # convert color from BGR to HSV
        imgs = rgb_to_hsv(imgs)

        # Apply saturation and hue
        for idx in range(batch_size):
            # random saturation
            if randint(idx, 2):
                imgs[idx, 1] *= uniform(idx, self.saturation_lower, self.saturation_upper)

            # random hue
            if randint(idx, 2):
                imgs[idx, 0] += uniform(idx, -self.hue_delta, self.hue_delta)

        imgs[:, 0][imgs[:, 0] > 360] -= 360
        imgs[:, 0][imgs[:, 0] < 0] += 360

        # convert color from HSV to BGR
        imgs = hsv_to_rgb(imgs)

        # Apply contrast mode 1 and channel swap
        for idx in range(batch_size):
            # random contrast
            if contrast_modes[idx] == 1:
                if randint(idx, 2):
                    alpha = uniform(idx, self.contrast_lower, self.contrast_upper)
                    imgs[idx] *= alpha

            # randomly swap channels
            if randint(idx, 2):
                imgs[idx] = imgs[idx, permutation(idx, 3)]

        imgs = imgs[:, [2, 1, 0], :, :]  # RGB to BGR

        return imgs



class DumpConfig:
    def __init__(self):
        self.enabled = True
        self.enabled = False

        # self.out_dir = tempfile.mkdtemp()
        self.out_dir = os.environ.get('STREAMRC_DEBUG_DIR', 'debug/debug_net')
        self.stage_count = 0
        self.frame_count = 0
        self.token=""
        self.file_path=""
    
    def add_token(self, token):
        self.frame_count += 1
        self.token = token

DUMP = DumpConfig()


# for backward compatibility
class Version:
    def __init__(self):
        self.name = 'v1.0.0'

VERSION = Version()
