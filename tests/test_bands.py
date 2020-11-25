# File under the MIT license, see https://github.com/adefossez/julius/LICENSE for details.
# Author: adefossez, 2020

import random
import unittest

import torch as th

from julius import SplitBands, split_bands
from julius.core import pure_tone


def delta(a, b, ref, fraction=0.9):
    length = a.shape[-1]
    compare_length = int(length * fraction)
    offset = (length - compare_length) // 2
    a = a[..., offset: offset + length]
    b = b[..., offset: offset + length]
    return 100 * th.abs(a - b).mean() / ref.std()


TOLERANCE = 0.5  # Tolerence to errors as percentage of the std of the input signal


class _BaseTest(unittest.TestCase):
    def assertSimilar(self, a, b, ref, msg=None, tol=TOLERANCE):
        self.assertLessEqual(delta(a, b, ref), tol, msg)


class TestLowPassFilters(_BaseTest):
    def setUp(self):
        th.manual_seed(1234)
        random.seed(1234)

    def test_keep_or_kill(self):
        sr = 256
        low = pure_tone(10, sr)
        mid = pure_tone(40, sr)
        high = pure_tone(100, sr)

        x = low + mid + high

        decomp = split_bands(x, sr, cutoffs=[20, 70])
        self.assertEqual(len(decomp), 3)
        for est, gt, name in zip(decomp, [low, mid, high], ["low", "mid", "high"]):
            self.assertSimilar(est, gt, gt, name)

    def test_torchscript(self):
        x = th.randn(128)

        mod = SplitBands(16, 4)
        jitted = th.jit.script(mod)
        self.assertEqual(list(jitted(x).shape), [4, 128])

    def test_repr(self):
        mod = SplitBands(16, 4, fft=False)
        self.assertEqual(repr(mod), "SplitBands(sample_rate=16,n_bands=4,fft=False)")


if __name__ == '__main__':
    unittest.main()
