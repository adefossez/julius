# File under the MIT license, see https://github.com/adefossez/julius/LICENSE for details.
# Author: adefossez, 2020

import argparse

from julius import lowpass_filter
from julius.core import pure_tone, volume
from julius.utils import Chrono, MarkdownTable


def test(table, freq, zeros, fft=None, device="cpu"):
    sr = 44_100

    attns = []
    for ratio in [0.9, 1, 1.1]:
        x = pure_tone(ratio * freq * sr, sr, 4, device=device)
        with Chrono() as chrono:
            y = lowpass_filter(x, freq, fft=fft, zeros=zeros)
        attns.append(format(volume(y) - volume(x), ".2f"))

    table.line([freq] + attns + [int(1000 * chrono.duration)])


def main():
    parser = argparse.ArgumentParser("lowpass.py")
    parser.add_argument("-d", "--device", default="cpu")
    parser.add_argument("--fft", action="store_true", default=None)
    parser.add_argument("--no_fft", action="store_false", dest="fft")
    parser.add_argument("--zeros", default=8, type=float)
    args = parser.parse_args()

    table = MarkdownTable(
        ["Freq.", "Attn. 0.9 (dB)", "Attn 1.0 (dB)", "Attn 1.1 (dB)", "Time (ms)"])
    table.header()
    for freq in [0.005, 0.01, 0.1, 0.2, 0.4]:
        test(table, freq, zeros=args.zeros, fft=args.fft, device=args.device)


if __name__ == "__main__":
    main()
