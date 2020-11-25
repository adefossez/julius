# File under the MIT license, see https://github.com/adefossez/julius/LICENSE for details.
# Author: adefossez, 2020

import random
import unittest

import torch as th
from torch.nn import functional as F

import julius

TOLERANCE = 1e-4  # as relative delta in percentage


class _BaseTest(unittest.TestCase):
    def setUp(self):
        th.manual_seed(1234)
        random.seed(1234)

    def assertSimilar(self, a, b, msg=None, tol=TOLERANCE):
        delta = 100 * th.norm(a - b) / th.norm(b)
        self.assertLessEqual(delta, tol, msg)

    def compare_pytorch(self, *args, block_ratio=10, msg=None, tol=TOLERANCE, **kwargs):
        y_ref = F.conv1d(*args, **kwargs)
        y = julius.fft_conv1d(*args, block_ratio=block_ratio, **kwargs)
        self.assertEqual(list(y.shape), list(y_ref.shape), msg)
        self.assertSimilar(y, y_ref, msg, tol)


class TestFFTConv1d(_BaseTest):
    def test_same_as_pytorch(self):
        for _ in range(5):
            kernel_size = random.randrange(4, 128)
            batch_size = random.randrange(1, 6)
            length = random.randrange(kernel_size, 1024)
            chin = random.randrange(1, 12)
            chout = random.randrange(1, 12)
            block_ratio = random.choice([5, 10, 20])
            bias = random.random() < 0.5
            if random.random() < 0.5:
                padding = 0
            else:
                padding = random.randrange(kernel_size // 2, 2 * kernel_size)
            x = th.randn(batch_size, chin, length)
            w = th.randn(chout, chin, kernel_size)
            keys = ["length", "kernel_size", "chin", "chout", "block_ratio", "bias"]
            loc = dict(locals())
            state = {key: loc[key] for key in keys}
            if bias:
                bias = th.randn(chout)
            else:
                bias = None
            for stride in [1, 2, 5]:
                state["stride"] = stride
                self.compare_pytorch(
                    x, w, bias, stride, padding, block_ratio=block_ratio,
                    msg=repr(state))

    def test_small_input(self):
        x = th.randn(1, 5, 19)
        w = th.randn(10, 5, 32)
        with self.assertRaises(RuntimeError):
            julius.fft_conv1d(x, w)

        x = th.randn(1, 5, 19)
        w = th.randn(10, 5, 19)
        self.assertEqual(list(julius.fft_conv1d(x, w).shape), [1, 10, 1])

    def test_block_ratio(self):
        x = th.randn(1, 5, 1024)
        w = th.randn(10, 5, 19)
        ref = julius.fft_conv1d(x, w)
        for block_ratio in [1, 5, 10, 20]:
            y = julius.fft_conv1d(x, w, block_ratio=block_ratio)
            self.assertSimilar(y, ref, msg=str(block_ratio))

        with self.assertRaises(RuntimeError):
            y = julius.fft_conv1d(x, w, block_ratio=0.9)

    def test_module(self):
        x = th.randn(16, 4, 1024)
        mod = julius.FFTConv1d(4, 5, 8, bias=True)
        mod(x)
        mod = julius.FFTConv1d(4, 5, 8, bias=False)
        mod(x)

    def test_torchscript(self):
        x = th.randn(16, 4, 1024)
        mod = julius.FFTConv1d(4, 5, 8, bias=True)
        jitted = th.jit.script(mod)
        self.assertEqual(list(jitted(x).shape), [16, 5, 1024 - 8 + 1])

    def test_repr(self):
        mod = julius.FFTConv1d(4, 5, 8, bias=False)
        self.assertEqual(
            repr(mod),
            "FFTConv1d(in_channels=4,out_channels=5,kernel_size=8,bias=False)")
