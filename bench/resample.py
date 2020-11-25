# File under the MIT license, see https://github.com/adefossez/julius/LICENSE for details.
# Author: adefossez, 2020

import argparse
import math

import resampy
import torch as th

from julius import resample_frac
from julius.utils import Chrono, MarkdownTable


def test(table, old_sr, new_sr, device="cpu"):
    x = th.randn(16, 8 * old_sr * int(math.ceil(44_100 / old_sr)), device=device)

    with Chrono() as chrono:
        y = resample_frac(x, old_sr, new_sr, zeros=56)
    dur_julius = int(1000 * chrono.duration)

    if device == "cpu":
        with Chrono() as chrono:
            y_resampy = th.from_numpy(resampy.resample(x.numpy(), old_sr, new_sr))
        dur_resampy = int(1000 * chrono.duration)

        delta = (y_resampy - y).abs().mean()
        table.line([old_sr, new_sr, dur_julius, dur_resampy, format(delta, ".1%")])
    else:
        table.line([old_sr, new_sr, dur_julius])


def main():
    parser = argparse.ArgumentParser("resample.py")
    parser.add_argument("-d", "--device", default="cpu")
    args = parser.parse_args()

    if args.device == "cpu":
        table = MarkdownTable(["Old sr", "New sr", "Julius (ms)", "Resampy (ms)", "Delta (%)"])
    else:
        table = MarkdownTable(["Old sr", "New sr", "Julius (ms)"])
    table.header()

    rates = [(2, 1), (1, 2), (4, 5), (10, 11), (44100, 16000), (20001, 30001)]
    for old_sr, new_sr in rates:
        test(table, old_sr, new_sr, device=args.device)


if __name__ == "__main__":
    main()
