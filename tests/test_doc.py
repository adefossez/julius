from doctest import testmod
import unittest

from julius import resample, fftconv, lowpass, bands, utils


class DocStringTest(unittest.TestCase):
    def test_resample(self):
        self.assertEqual(testmod(resample).failed, 0)

    def test_fftconv(self):
        self.assertEqual(testmod(fftconv).failed, 0)

    def test_lowpass(self):
        self.assertEqual(testmod(lowpass).failed, 0)

    def test_bands(self):
        self.assertEqual(testmod(bands).failed, 0)

    def test_utils(self):
        self.assertEqual(testmod(utils).failed, 0)
