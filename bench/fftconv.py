# File under the MIT license, see https://github.com/adefossez/julius/LICENSE for details.
# Author: adefossez, 2020

import argparse

import torch as th
from torch.nn import functional as F

from julius import fft_conv1d
from julius.utils import Chrono, MarkdownTable


def test(table, kernel_size, block_ratio=5, device=None):
    x = th.randn(32, 32, 1024 * 10).to(device)
    w = th.randn(64, 32, kernel_size).to(device)

    with Chrono() as chrono_ref:
        y_ref = F.conv1d(x, w)

    with Chrono() as chrono_fft:
        y_fft = fft_conv1d(x, w, block_ratio=block_ratio)

    delta = format((y_ref - y_fft).abs().mean(), ".1e")
    table.line(
        [kernel_size,  int(1000 * chrono_fft.duration), int(1000 * chrono_ref.duration), delta])


def main():
    parser = argparse.ArgumentParser("fftconv.py")
    parser.add_argument("-d", "--device", default="cpu")
    parser.add_argument("-b", "--block_ratio", default=5, type=float)
    args = parser.parse_args()

    table = MarkdownTable(
        ["Kernel size", "FFT (ms)", "No FFT (ms)", "  Delta"])
    table.header()

    for kernel_size in [8, 32, 64, 128, 256, 1024, 2048]:
        test(table, kernel_size, block_ratio=args.block_ratio, device=args.device)


if __name__ == "__main__":
    main()
