# File under the MIT license, see https://github.com/adefossez/julius/LICENSE for details.
# Author: adefossez, 2020

import math
import random
import unittest

import resampy
import torch as th

from julius import resample


def pure_tone(freq, sr=128, dur=4):
    time = th.arange(sr * dur).float() / sr
    return th.cos(2 * math.pi * freq * time)


def delta(a, b, ref, fraction=0.8):
    length = a.shape[-1]
    compare_length = int(length * fraction)
    offset = (length - compare_length) // 2
    a = a[..., offset: offset + length]
    b = b[..., offset: offset + length]
    return 100 * th.abs(a - b).mean() / ref.std()


TOLERANCE = 1  # Tolerence to errors as percentage of the std of the input signal


class _BaseTest(unittest.TestCase):
    def assertSimilar(self, a, b, ref, msg=None, tol=TOLERANCE):
        self.assertLessEqual(delta(a, b, ref), tol, msg)


class TestDownsample2(_BaseTest):
    def test_lowfreqs(self):
        # For those freq, downsample2 should be almost decimation
        for freq in [8, 16, 20, 28]:
            x = pure_tone(freq, sr=128)
            y_gt = x[::2]
            y = resample._downsample2(x)
            self.assertSimilar(y, y_gt, x, f"freq={freq}")

    def test_hifreqs(self):
        # For those freq, downsample2 should return zero
        for freq in [36, 40, 56, 64]:
            x = pure_tone(freq, sr=128)
            y = resample._downsample2(x)
            y_gt = 0 * y
            self.assertSimilar(y, y_gt, x, f"freq={freq}")

    def test_mixture(self):
        # Test one mixture
        x_low = pure_tone(16, sr=128)
        x_high = pure_tone(40, sr=128)
        x = x_low + x_high
        y = resample._downsample2(x)
        y_gt = x_low[::2]
        self.assertSimilar(y, y_gt, x, "mixture")


class TestUpsample2(_BaseTest):
    def test_upsample(self):
        # For those freq, _downsample2 should be almost decimation
        for freq in [8, 16, 20, 28, 32]:
            x = pure_tone(freq, sr=64)
            y_gt = pure_tone(freq, sr=128)
            y = resample._upsample2(x)
            self.assertSimilar(y, y_gt, x, f"freq={freq}")


class TestResampleFrac(_BaseTest):
    def test_ref(self):
        # Compare to _upsample2 and _downsample2
        for freq in [8, 16, 20, 28, 32, 36, 40, 56, 64]:
            x = pure_tone(freq, sr=128)
            y_gt_down = resample._downsample2(x)
            y_down = resample.resample_frac(x, 2, 1, rolloff=1)
            self.assertSimilar(y_down, y_gt_down, x, f"freq={freq} down")
            y_gt_up = resample._upsample2(x)
            y_up = resample.resample_frac(x, 1, 2, rolloff=1)
            self.assertSimilar(y_up, y_gt_up, x, f"freq={freq} up")

    def test_resampy(self):
        old_sr = 3
        new_sr = 2
        x = pure_tone(7, sr=128, dur=3) + pure_tone(24, sr=128, dur=3)
        y_re = th.from_numpy(resampy.resample(x.numpy(), old_sr, new_sr)).float()
        y = resample.resample_frac(x, old_sr, new_sr)
        self.assertSimilar(y, y_re, x, f"{old_sr} to {new_sr}")

        old_sr = 2
        new_sr = 5
        x = pure_tone(7, sr=128) + pure_tone(48, sr=128)
        y_re = th.from_numpy(resampy.resample(x.numpy(), old_sr, new_sr)).float()
        y = resample.resample_frac(x, old_sr, new_sr)
        self.assertSimilar(y, y_re, x, f"{old_sr} to {new_sr}")

        random.seed(1234)
        th.manual_seed(1234)
        for _ in range(10):
            old_sr = random.randrange(8, 128)
            new_sr = random.randrange(8, 128)
            x = th.randn(1024)
            y_re = th.from_numpy(resampy.resample(x.numpy(), old_sr, new_sr)).float()
            y = resample.resample_frac(x, old_sr, new_sr, zeros=56)
            # We allow some relatively high tolerance as we are not using the same window.
            self.assertSimilar(y, y_re, x, f"{old_sr} to {new_sr}", tol=2)

    def test_torchscript(self):
        mod = resample.ResampleFrac(5, 7)
        x = th.randn(5 * 26)
        jitted = th.jit.script(mod)
        self.assertEqual(list(jitted(x).shape), [7 * 26])


if __name__ == '__main__':
    unittest.main()
