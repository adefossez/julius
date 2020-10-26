import math
import unittest
import sys

try:
    import resampy
except ImportError:
    print("Could not import resampy, skipping comparison tests.", file=sys.stderr)
    resampy = None


import torch as th

import julius


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
            y = julius.downsample2(x)
            self.assertSimilar(y, y_gt, x, f"freq={freq}")

    def test_hifreqs(self):
        # For those freq, downsample2 should return zero
        for freq in [36, 40, 56, 64]:
            x = pure_tone(freq, sr=128)
            y = julius.downsample2(x)
            y_gt = 0 * y
            self.assertSimilar(y, y_gt, x, f"freq={freq}")

    def test_mixture(self):
        # Test one mixture
        x_low = pure_tone(16, sr=128)
        x_high = pure_tone(40, sr=128)
        x = x_low + x_high
        y = julius.downsample2(x)
        y_gt = x_low[::2]
        self.assertSimilar(y, y_gt, x, "mixture")


class TestUpsample2(_BaseTest):
    def test_upsample(self):
        # For those freq, downsample2 should be almost decimation
        for freq in [8, 16, 20, 28, 32]:
            x = pure_tone(freq, sr=64)
            y_gt = pure_tone(freq, sr=128)
            y = julius.upsample2(x)
            self.assertSimilar(y, y_gt, x, f"freq={freq}")


class TestResample(_BaseTest):
    def test_ref(self):
        # Compare to upsample2 and downsample2
        for freq in [8, 16, 20, 28, 32, 36, 40, 56, 64]:
            x = pure_tone(freq, sr=128)
            y_gt_down = julius.downsample2(x)
            y_down = julius.resample_frac(x, 2, 1)
            self.assertSimilar(y_down, y_gt_down, x, f"freq={freq} down")
            y_gt_up = julius.upsample2(x)
            y_up = julius.resample_frac(x, 1, 2)
            self.assertSimilar(y_up, y_gt_up, x, f"freq={freq} up")

    def test_resampy(self):
        if resampy is None:
            return
        old_sr = 3
        new_sr = 2
        x = pure_tone(7, sr=128, dur=3) + pure_tone(24, sr=128, dur=3)
        y_re = th.from_numpy(resampy.resample(x.numpy(), old_sr, new_sr)).float()
        y = julius.resample_frac(x, old_sr, new_sr)
        self.assertSimilar(y, y_re, x, f"{old_sr} to {new_sr}")

        old_sr = 2
        new_sr = 5
        x = pure_tone(7, sr=128) + pure_tone(48, sr=128)
        y_re = th.from_numpy(resampy.resample(x.numpy(), old_sr, new_sr)).float()
        y = julius.resample_frac(x, old_sr, new_sr)
        self.assertSimilar(y, y_re, x, f"{old_sr} to {new_sr}")


if __name__ == '__main__':
    unittest.main()
