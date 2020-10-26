# File under the MIT license, see https://github.com/adefossez/julius/LICENSE for details.
# Author: adefossez, 2020
"""
Differentiable, Pytorch based resampling.
Implementation of Julius O. Smith algorithm for resampling.
See https://ccrma.stanford.edu/~jos/resample/ for details.
This implementation is specially optimized for when new_sr / old_sr is a fraction
with a small numerator and denominator when removing the gcd.
For instance if `new_sr / old_sr = 12 / 10` it will be fast, but not if
`new_sr / old_sr = 10001 / 10000`.

Very similar to bmcfee/resampy except this implementation is optimized
for the case mentioned before, while resampy is slower but more general.

You probably want to use `resample_frac`, although for the special case of resampling by
a factor of 2, you can also use `downsample2`
"""

import math

import torch as th
from torch.nn import functional as F


def sinc(t):
    """
    Implementation of sinc, i.e. sin(x) / x
    """
    return th.where(t == 0, th.tensor(1., device=t.device, dtype=t.dtype), th.sin(t) / t)


def resample_frac(x, old_sr: int, new_sr: int, zeros: int = 24, rolloff: float = 1):
    """
    Resampling from the sample rate `old_sr` to `new_sr`.
    After dividing `old_sr` and `new_sr` by their GCD, both should be small
    for this implementation to be fast.
    Args:
        x (Tensor): signal to resample, time should be the last dimension
        old_sr (int): sample rate of the input signal x
        new_sr (int): sample rate of the output.
        zeros (int): number of zero crossing to keep in the sinc filter.
        rolloff (float): when downsampling, use a lowpass filter that is `rolloff * new_sr / 2`,
            to ensure sufficient margin due to the imperfection of the FIR filter used.
            Default is deactivated, set it to a value < 1 if you experience aliasing issues.
    """
    *other, length = x.shape
    x = x.reshape(-1, length)

    gcd = math.gcd(old_sr, new_sr)
    old_sr //= gcd
    new_sr //= gcd

    kernels = []
    sr = min(new_sr, old_sr)
    if new_sr < old_sr:
        sr *= rolloff
    width = math.ceil(zeros * old_sr / sr)
    idx = th.arange(-width, width + old_sr).to(x)
    for i in range(new_sr):
        t = (-i/new_sr + idx/old_sr) * sr
        t = t.clamp_(-zeros, zeros)
        t *= math.pi
        window = th.cos(t/zeros/2)**2
        kernels.append(sinc(t) * window)

    scale = sr / old_sr
    kernel = th.stack(kernels).view(new_sr, 1, -1).mul_(scale)
    x = F.pad(x[:, None], (width, width + old_sr))
    ys = F.conv1d(x, kernel, stride=old_sr)
    return ys.transpose(1, 2).reshape(*other, -1)[..., :int(new_sr * length / old_sr)]


# Easier implementations for downsampling and upsampling by a factor of 2
# Kept for testing and reference

def _kernel_upsample2(zeros):
    win = th.hann_window(4 * zeros + 1, periodic=False)
    winodd = win[1::2]
    t = th.linspace(-zeros + 0.5, zeros - 0.5, 2 * zeros)
    t *= math.pi
    kernel = (sinc(t) * winodd).view(1, 1, -1)
    return kernel


def upsample2(x, zeros=24):
    """
    Upsample x by a factor of two. The output will be exactly twice as long as the input.
    Args:
        x (Tensor): signal to upsample, time should be the last dimension
        zeros (int): number of zero crossing to keep in the sinc filter.
    """
    *other, time = x.shape
    kernel = _kernel_upsample2(zeros).to(x)
    out = F.conv1d(x.view(-1, 1, time), kernel, padding=zeros)[..., 1:].view(*other, time)
    y = th.stack([x, out], dim=-1)
    return y.view(*other, -1)


def _kernel_downsample2(zeros):
    win = th.hann_window(4 * zeros + 1, periodic=False)
    winodd = win[1::2]
    t = th.linspace(-zeros + 0.5, zeros - 0.5, 2 * zeros)
    t.mul_(math.pi)
    kernel = (sinc(t) * winodd).view(1, 1, -1)
    return kernel


def downsample2(x, zeros=24):
    """
    Downsample x by a factor of two. The output length is half of the input, ceiled.
    Args:
        x (Tensor): signal to downsample, time should be the last dimension
        zeros (int): number of zero crossing to keep in the sinc filter.
    """
    if x.shape[-1] % 2 != 0:
        x = F.pad(x, (0, 1))
    xeven = x[..., ::2]
    xodd = x[..., 1::2]
    *other, time = xodd.shape
    kernel = _kernel_downsample2(zeros).to(x)
    out = xeven + F.conv1d(xodd.view(-1, 1, time), kernel, padding=zeros)[..., :-1].view(
        *other, time)
    return out.view(*other, -1).mul(0.5)
