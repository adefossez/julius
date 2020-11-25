## Benchmarking and verification of Julius

In order to verify the correctness and speed of the implementations in Julius,
we compare ourselves to different reference implementations, comparing speed and
checking how far we are.

### ResampleFrac

We compare `julius.resample` to `resampy`, on an input of size (32, 8 * 44100),
i.e. a batch of size 16 of 8 second of audio at 44.1kHz.
We use the same number of zero crossing as `resampy` for this benchmark.
The small delta is probably
due to the different window function used.


On CPU we have:

| Old sr | New sr | Julius (ms) | Resampy (ms) | Delta (%) |
|--------|--------|-------------|--------------|-----------|
|      2 |      1 |         288 |         1530 |      1.3% |
|      1 |      2 |         367 |         2329 |      1.8% |
|      4 |      5 |          91 |         1439 |      1.8% |
|     10 |     11 |          54 |         1260 |      1.8% |
|  44100 |  16000 |          62 |         2180 |      1.0% |
|  20001 |  30001 |       19922 |         2313 |      1.8% |


On GPU we have:

Not available /!\

### FFTConv1d

We compare to `pytorch.nn.functional.conv1d`, on a input of size [16, 16, 10240],
for a convolution with 16 input channels, 32 output channels and various kernel sizes.

On CPU we have:

| Kernel size | Stride | FFT (ms) | No FFT (ms) |   Delta |
|-------------|--------|----------|-------------|---------|
|           8 |      1 |      123 |          32 | 2.3e-06 |
|           8 |      4 |       72 |           8 | 2.3e-06 |
|          32 |      1 |       86 |          51 | 7.4e-06 |
|          32 |      4 |       78 |          17 | 7.3e-06 |
|          64 |      1 |       79 |          97 | 1.4e-05 |
|          64 |      4 |       81 |          24 | 1.4e-05 |
|         128 |      1 |       91 |         183 | 2.7e-05 |
|         128 |      4 |       91 |          52 | 2.7e-05 |
|         256 |      1 |      119 |         426 | 5.3e-05 |
|         256 |      4 |      123 |         124 | 5.3e-05 |
|        1024 |      1 |      245 |        1885 | 2.1e-04 |
|        1024 |      4 |      226 |         509 | 2.1e-04 |
|        2048 |      1 |      294 |        3336 | 4.1e-04 |
|        2048 |      4 |      301 |         909 | 4.1e-04 |


On GPU we have:

Not available /!\

### LowPassFilter

We do not compare to anything, but measure the attenuation in dB of a pure tone
at `0.9 * cutoff`, at the `cutoff`, and at `1.1 * cutoff`.
Note that our implementation automatically choses to use FFTConv1d or not when appropriate.

On CPU we have:

| Freq. | Attn. 0.9 (dB) | Attn 1.0 (dB) | Attn 1.1 (dB) | Time (ms) |
|-------|----------------|---------------|---------------|-----------|
| 0.005 |          -1.41 |         -6.02 |        -16.44 |         6 |
|  0.01 |          -1.41 |         -6.02 |        -16.46 |         6 |
|   0.1 |          -1.41 |         -6.02 |        -16.48 |         6 |
|   0.2 |          -1.41 |         -6.02 |        -16.48 |        10 |
|   0.4 |          -1.41 |         -6.03 |        -16.38 |         6 |


On GPU we have:

Not available /!\


