# Julius, fast PyTorch based resampling for audio and 1D signals

![tests badge](https://github.com/adefossez/julius/workflows/tests/badge.svg)

This is an implementation of the [sinc resample algorithm][resample] by Julius O. Smith.
It is the same algorithm than the one used in [resampy][resampy] but to run efficiently on GPU it
is limited to fractional changes of the sample rate. It will be fast if the old and new sample rate
are small after dividing them by their GCD. For instance going from a sample rate of 2000 to 3000 (2, 3 after removing the GCD)
will be extremely fast, while going from 20001 to 30001 will not.

Julius is faster than resampy even on CPU, and when running on GPU it makes resampling a completely negligible part of your pipeline.
Finally, Julius is differentiable and can thus be integrated in an end-to-end training pipeline.

<p align="center">
<img src="./logo.png" alt="Representation of the convolutions filters used for the efficient resampling."
width="500px"></p>

## Installation

`julius` requires python 3.6. To install:
```bash
pip3 install julius
```

## Usage

```python
import julius
julius.resample_frac(signal, old_sr, new_sr, zeros=24, rollof=1)
```

- `signal` is a multi dimensional PyTorch tensor, with the last dimension representing time.
- `resample_frac` change the sample rate from `old_sr` to `new_sr`. The GCD is automatically removed for you.
- `zeros` is the number of zero crossing to keep in the sinc filters, higher values can be more accurate but also slower. Default value is probably fine.
- If `rolloff < 1`, the cutoff frequency of the low pass filter used before downsampling will be half the target sample_rate times this amount. This can potentially reduce aliasing if you notice such an issue. When doing upsampling, this is ignored.

If `signal` is a CUDA Tensor, then everything will run on GPU :)

## Benchmark

On my laptop, the time taken to process a tensor of size `(256, 40000)` (roughly 256 seconds of audio at 44.1 kHz) is (on CPU):

| Old sr | New sr | Julius (sec) | Resampy (sec) |
|--------|--------|--------|---------|
|       2|       1|   0.4  |2.0 |
| 1 | 2 | 0.6 | 4.8 |
| 4 | 5 | 0.13 | 2.5|
| 10 | 11 | 0.08 | 2.45 |
| 20001 | 30001 | 2.4 | 1.5 |

Except when `new_sr / old_sr` does not simplify to a small irreductible fraction, `julius` is faster even on CPU than `resampy`.
When running on GPU, `julius` makes resampling take a negligible time of the order of a few milliseconds.


## Running tests

Clone this repository, then
```bash
pip3 install .[test]'
python3 tests.py
```

## License

`julius` is released under the MIT license.


[resample]: https://ccrma.stanford.edu/~jos/resample/resample.html
[resampy]: https://resampy.readthedocs.io/
