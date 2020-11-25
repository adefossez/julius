# Julius, fast PyTorch based DSP for audio and 1D signals

![tests badge](https://github.com/adefossez/julius/workflows/tests/badge.svg)
![cov badge](https://github.com/adefossez/julius/workflows/cov%3E90%25/badge.svg)

Julius contains different Digital Signal Processing algorithms implemented
with PyTorch, so that they are differentiable and available on CUDA.
Note that all the modules implemented here can be used with TorchScript.

For now, I have implemented:

- [julius.resample](https://adefossez.github.io/julius/julius/resample.html): fast sinc resampling.
- [julius.fftconv](https://adefossez.github.io/julius/julius/fftconv.html): FFT based convolutions.
- [julius.lowpass](https://adefossez.github.io/julius/julius/lowpass.html): FIR low pass filter banks.
- [julius.bands](https://adefossez.github.io/julius/julius/bands.html): Decomposition of a waveform signal over mel-scale frequency bands.

Along that, you might found useful utilities in:

- [julius.core](https://adefossez.github.io/julius/julius/core.html): DSP related functions.
- [julius.utils](https://adefossez.github.io/julius/julius/utils.html): Generic utilities.

<p align="center">
<img src="./logo.png" alt="Representation of the convolutions filters used for the efficient resampling."
width="500px"></p>

## Installation

`julius` requires python 3.6. To install:
```bash
pip3 install -U julius
```


## Usage

See the [Julius documentation][docs] for the usage of Julius. Hereafter you will find a few examples
to get you quickly started:

```python3
import julius
import torch

signal = torch.randn(6, 4, 1024)
# Resample from a sample rate of 100 to 70. The old and new sample rate must be integers, 
# and resampling will be fast if they form an irreductible fraction with small numerator 
# and denominator (here 10 and 7). Any shape is supported, last dim is time.
resampled_signal = julius.resample_frac(signal, 100, 70)

# Low pass filter with a `0.1 * sample_rate` cutoff frequency.
low_freqs = julius.lowpass_filter(signal, 0.1)

# Fast convolutions with FFT, useful for large kernels
conv = julius.FFTConv1d(4, 10, 512)
convolved = conv(signal)

# Decomposition over frequency bands in the Waveform domain
bands = julius.split_bands(signal, n_bands=10, sample_rate=100)
# Decomposition with n_bands frequency bands evenly spaced in mel space.
# Input shape can be `[*, T]`, output will be `[n_bands, *, T]`.
random_eq = (torch.rand(10, 1, 1, 1) * bands).sum(0)
```

## Algorithms

### Resample

This is an implementation of the [sinc resample algorithm][resample] by Julius O. Smith.
It is the same algorithm than the one used in [resampy][resampy] but to run efficiently on GPU it
is limited to fractional changes of the sample rate. It will be fast if the old and new sample rate
are small after dividing them by their GCD. For instance going from a sample rate of 2000 to 3000 (2, 3 after removing the GCD)
will be extremely fast, while going from 20001 to 30001 will not.
Julius resampling is faster than resampy even on CPU, and when running on GPU it makes resampling a completely negligible part of your pipeline
(except of course for weird cases like going from a sample rate of 20001 to 30001).


### FFTConv1d

Computing convolutions with very large kernels (>= 128) and a stride of 1 can be much faster
using FFT. This implements the same API as `torch.nn.Conv1d` and `torch.nn.functional.conv1d`
but with a FFT backend. Dilation and groups are not supported.
FFTConv will be faster on CPU even for relatively small tensors (a few dozen channels, kernel size
of 128). On CUDA, due to the higher parallelism, regular convolution can be faster in many cases,
but for kernel sizes above 128, for a large number of channels or batch size, FFTConv1d
will eventually be faster (basically when you no longer have idle cores that can hide
the true complexity of the operation).

### LowPass

Classical Finite Impulse Reponse windowed sinc lowpass filter. It will use FFT convolutions automatically
if the filter size is large enough.

### Bands

Decomposition of a signal over frequency bands in the waveform domain. This can be useful for
instance to perform parametric EQ (see [Usage](#usage) above).

## Benchmarks

You can find speed tests (and comparisons to reference implementations) on the
[benchmark][bench]. The CPU benchmarks are run on a Mac Book Pro 2020, with a 2 GHz
quadcore intel CPU. The GPUs benchmark are run on Google Colab Pro (e.g. V100 or P100 NVidia GPU).
We also compare the validity of our implementations, as compared to reference ones like `resampy`
or `torch.nn.Conv1d`.



## Running tests

Clone this repository, then
```bash
pip3 install .[dev]'
python3 tests.py
```

To run the benchmarks:
```
pip3 install .[dev]'
python3 -m bench.gen
```


## License

`julius` is released under the MIT license.

## Thanks

This package is named in the honor of
[Julius O. Smith](https://ccrma.stanford.edu/~jos/),
whose books and website were a gold mine of information for me to learn about DSP. Go checkout his website if you want
to learn more about DSP.


[resample]: https://ccrma.stanford.edu/~jos/resample/resample.html
[resampy]: https://resampy.readthedocs.io/
[docs]:  https://adefossez.github.io/julius/julius/index.html
[bench]:  ./bench.md
