# Julius, fast PyTorch based resampling for audio and 1D signals

This is an implementation of the [sinc resample algorithm][resample] by Julius O. Smith.
It is the same algorithm than the one used in [resampy][resampy] but to run efficiently on GPU it
is limited to fractional changes of the sample rate. It will be fast if the old and new sample rate
are small after dividing them by their GCD. For instance going from a sample rate of 2000 to 3000 (2, 3 after removing the GCD)
will be extremely fast, while going from 2001 to 3000 will not.

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

`signal` is a multi dimensional PyTorch tensor, with the last dimension representing time.
`resample_frac` change the sample rate from `old_sr` to `new_sr`. The GCD is automatically removed for you.
`zeros` is the number of zero crossing to keep in the sinc filters, higher values can be more accurate but also slower. Default value is probably fine.
If `rolloff < 1`, the cutoff frequency of the low pass filter used before downsampling will be lower than half of the target sample_rate
by this amount. This can potentially reduce aliasing if you notice such an issue.

If `signal` is a CUDA Tensor, then everything will run on GPU :)

## Benchmark


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
