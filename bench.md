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
|      2 |      1 |         328 |         1317 |      1.3% |
|      1 |      2 |         385 |         1884 |      1.8% |
|      4 |      5 |          92 |         1191 |      1.8% |
|     10 |     11 |          54 |         1088 |      1.8% |
|  44100 |  16000 |          57 |         2176 |      0.9% |
|  20001 |  30001 |       19243 |         2113 |      1.8% |


On GPU we have:

Not available /!\

### FFTConv1d

We compare to `pytorch.nn.functional.conv1d`, on a input of size [16, 16, 10240],
for a convolution with 16 input channels, 32 output channels and various kernel sizes.

On CPU we have:

| Kernel size | FFT (ms) | No FFT (ms) |   Delta |
|-------------|----------|-------------|---------|
|           8 |      116 |          39 | 2.2e-06 |
|          32 |       74 |          60 | 7.6e-06 |
|          64 |       63 |         114 | 1.4e-05 |
|         128 |       64 |         218 | 2.7e-05 |
|         256 |       73 |         424 | 5.3e-05 |
|        1024 |      188 |        1926 | 2.1e-04 |
|        2048 |      260 |        2988 | 4.1e-04 |


On GPU we have:

Not available /!\

### LowPassFilter

We do not compare to anything, but measure the attenuation in dB of a pure tone
at `0.9 * cutoff`, at the `cutoff`, and at `1.1 * cutoff`.
Note that our implementation automatically choses to use FFTConv1d or not when appropriate.

On CPU we have:

| Freq. | Attn. 0.9 (dB) | Attn 1.0 (dB) | Attn 1.1 (dB) | Time (ms) |
|-------|----------------|---------------|---------------|-----------|
| 0.005 |          -1.41 |         -6.02 |        -16.44 |         7 |
|  0.01 |          -1.41 |         -6.02 |        -16.46 |         6 |
|   0.1 |          -1.41 |         -6.02 |        -16.48 |         9 |
|   0.2 |          -1.41 |         -6.02 |        -16.48 |         5 |
|   0.4 |          -1.41 |         -6.03 |        -16.38 |         5 |


On GPU we have:

Not available /!\


