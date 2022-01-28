# File under the MIT license, see https://github.com/adefossez/julius/LICENSE for details.
# Author: adefossez, 2020
import os
import tempfile
import uuid
from pathlib import Path

import math
import random
import unittest

import resampy
import torch as th

from julius import resample, ResampleFrac

is_onnxruntime_installed = True
try:
    import onnxruntime
except ImportError:
    print("Warning: onnxruntime is not installed. Some tests may be skipped")
    is_onnxruntime_installed = False


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


TOLERANCE = 1  # Tolerance to errors as percentage of the std of the input signal


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

    def test_constant(self):
        x = th.ones(4096)
        for zeros in [4, 10]:
            for old_sr in [1, 4, 10]:
                for new_sr in [1, 4, 10]:
                    y_low = resample.resample_frac(x, old_sr, new_sr, zeros=zeros)
                    self.assertLessEqual(
                        (y_low - 1).abs().mean(), 1e-6, (zeros, old_sr, new_sr))

    def test_default_output_length(self):
        x = th.ones(1, 2, 32000)

        resampler = resample.ResampleFrac(old_sr=32000, new_sr=48000)
        y = resampler(x)
        self.assertEqual(y.shape, (1, 2, 48000))

        # Test functional version as well
        y = resample.resample_frac(x, old_sr=32000, new_sr=48000)
        self.assertEqual(y.shape, (1, 2, 48000))

    def test_custom_output_length(self):
        x = th.ones(1, 32001)

        resampler = resample.ResampleFrac(old_sr=32000, new_sr=48000)
        y = resampler(x, output_length=48001)
        self.assertEqual(y.shape, (1, 48001))

        # Test functional version as well
        y = resample.resample_frac(x, old_sr=32000, new_sr=48000, output_length=47999)
        self.assertEqual(y.shape, (1, 47999))

    def test_custom_output_length_extreme_resampling(self):
        """
        Resample a signal from 1 hz to 499 hz to check that custom_length works
        correctly without extra internal padding
        """
        x = th.ones(1, 1)

        resampler = resample.ResampleFrac(old_sr=1, new_sr=499)
        y = resampler(x, output_length=499)
        self.assertEqual(y.shape, (1, 499))

        # Test functional version as well
        y = resample.resample_frac(x, old_sr=1, new_sr=499, output_length=3)
        self.assertEqual(y.shape, (1, 3))

    def test_custom_output_length_out_of_range(self):
        x = th.ones(1, 32000)
        with self.assertRaisesRegex(
            ValueError, "output_length must be between 0 and 48000"
        ):
            resample.resample_frac(x, old_sr=32000, new_sr=48000, output_length=48002)

    def test_full(self):
        x = th.randn(19)
        y = resample.resample_frac(x, 7, 1, full=True)
        self.assertEqual(len(y), 3)
        z = resample.resample_frac(y, 5, 1, full=True)
        y2 = resample.resample_frac(z, 1, 5, full=True)
        x2 = resample.resample_frac(y2, 1, 7, output_length=len(x))
        self.assertEqual(x.shape, x2.shape)

    @unittest.skipUnless(is_onnxruntime_installed, "onnxruntime is not installed")
    def test_onnx_compatibility(self):
        tmp_onnx_file_path = os.path.join(
            tempfile.gettempdir(), str(uuid.uuid4()) + ".onnx"
        )
        try:
            resampler = ResampleFrac(old_sr=32_000, new_sr=16_000)
            example_input1 = th.rand(1, 100, dtype=th.float32)
            example_input2 = th.rand(1, 124, dtype=th.float32)

            th.onnx.export(
                resampler,
                example_input1,
                tmp_onnx_file_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={
                    "input": {0: "num_channels", 1: "num_samples"},
                    "output": {0: "num_channels", 1: "num_samples"},
                },
            )
            onnx_model = onnxruntime.InferenceSession(tmp_onnx_file_path)
            onnxruntime_output = onnx_model.run(
                ["output"], {"input": example_input2.numpy()}
            )[0]
            self.assertEqual(onnxruntime_output.shape[-1], 62)

            torch_output = resampler(example_input2)
            self.assertEqual(torch_output.shape[-1], 62)

            self.assertSimilar(
                th.from_numpy(onnxruntime_output), torch_output, example_input2
            )
        finally:
            if os.path.isfile(tmp_onnx_file_path):
                os.remove(tmp_onnx_file_path)


if __name__ == '__main__':
    unittest.main()
