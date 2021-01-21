# File under the MIT license, see https://github.com/adefossez/julius/LICENSE for details.
# Author: adefossez, 2020
"""
Signal processing related utilities.
"""
import math
import torch
from torch.nn import functional as F


def sinc(x: torch.Tensor):
    """
    Implementation of sinc, i.e. sin(x) / x

    __Warning__: the input is not multiplied by `pi`!
    """
    return torch.where(x == 0, torch.tensor(1., device=x.device, dtype=x.dtype), torch.sin(x) / x)


def pad_to(tensor: torch.Tensor, target_length: int, mode: str = 'constant', value: float = 0):
    """
    Pad the given tensor to the given length, with 0s on the right.
    """
    return F.pad(tensor, (0, target_length - tensor.shape[-1]), mode=mode, value=value)


def hz_to_mel(freqs: torch.Tensor):
    """
    Converts a Tensor of frequencies in hertz to the mel scale.
    Uses the simple formula by O'Shaughnessy (1987).

    Args:
        freqs (torch.Tensor): frequencies to convert.

    """
    return 2595 * torch.log10(1 + freqs / 700)


def mel_to_hz(mels: torch.Tensor):
    """
    Converts a Tensor of mel scaled frequencies to Hertz.
    Uses the simple formula by O'Shaughnessy (1987).

    Args:
        mels (torch.Tensor): mel frequencies to convert.
    """
    return 700 * (10**(mels / 2595) - 1)


def mel_frequencies(n_mels: int, fmin: float, fmax: float):
    """
    Return frequencies that are evenly spaced in mel scale.

    Args:
        n_mels (int): number of frequencies to return.
        fmin (float): start from this frequency (in Hz).
        fmax (float): finish at this frequency (in Hz).


    """
    low = hz_to_mel(torch.tensor(float(fmin))).item()
    high = hz_to_mel(torch.tensor(float(fmax))).item()
    mels = torch.linspace(low, high, n_mels)
    return mel_to_hz(mels)


def volume(x: torch.Tensor, floor=1e-8):
    """
    Return the volume in dBFS.
    """
    return torch.log10(floor + (x**2).mean(-1)) * 10


def pure_tone(freq: float, sr: float = 128, dur: float = 4, device=None):
    """
    Return a pure tone, i.e. cosine.

    Args:
        freq (float): frequency (in Hz)
        sr (float): sample rate (in Hz)
        dur (float): duration (in seconds)
    """
    time = torch.arange(int(sr * dur), device=device).float() / sr
    return torch.cos(2 * math.pi * freq * time)
