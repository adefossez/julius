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
|      2 |      1 |         317 |         1435 |      1.2% |
|      1 |      2 |         410 |         2131 |      1.8% |
|      4 |      5 |          98 |         1391 |      1.8% |
|     10 |     11 |          53 |         1241 |      1.8% |
|  44100 |  16000 |          68 |         2554 |      0.9% |
|  20001 |  30001 |       21370 |         2249 |      1.8% |


On GPU we have:

| Old sr | New sr | Julius (ms) |
|--------|--------|-------------|
|      2 |      1 |          22 |
|      1 |      2 |           9 |
|      4 |      5 |           6 |
|     10 |     11 |           6 |
|  44100 |  16000 |          27 |
|  20001 |  30001 |       14974 |

### FFTConv1d

We compare to `pytorch.nn.functional.conv1d`, on a input of size [32, 32, 10240],
for a convolution with 32 input channels, 64 output channels and various kernel sizes.

On CPU we have:

| Kernel size | FFT (ms) | No FFT (ms) |   Delta |
|-------------|----------|-------------|---------|
|           8 |      508 |         208 | 4.0e-06 |
|          32 |      327 |         454 | 1.4e-05 |
|          64 |      322 |         885 | 2.7e-05 |
|         128 |      292 |        1714 | 5.3e-05 |
|         256 |      317 |        3355 | 1.0e-04 |
|        1024 |      799 |       13821 | 4.1e-04 |
|        2048 |      965 |       24802 | 8.3e-04 |


On GPU we have:

| Kernel size | FFT (ms) | No FFT (ms) |   Delta |
|-------------|----------|-------------|---------|
|           8 |       19 |           4 | 4.5e-06 |
|          32 |       24 |           6 | 1.5e-05 |
|          64 |       32 |          12 | 3.1e-05 |
|         128 |       36 |          24 | 5.4e-05 |
|         256 |       37 |          48 | 1.1e-04 |
|        1024 |       84 |         378 | 4.2e-04 |
|        2048 |       70 |         642 | 8.3e-04 |


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
|   0.4 |          -1.41 |         -6.03 |        -16.38 |         4 |


On GPU we have:

| Freq. | Attn. 0.9 (dB) | Attn 1.0 (dB) | Attn 1.1 (dB) | Time (ms) |
|-------|----------------|---------------|---------------|-----------|
| 0.005 |          -1.41 |         -6.02 |        -16.44 |         3 |
|  0.01 |          -1.41 |         -6.02 |        -16.46 |         2 |
|   0.1 |          -1.41 |         -6.02 |        -16.48 |         1 |
|   0.2 |          -1.41 |         -6.02 |        -16.48 |         0 |
|   0.4 |          -1.41 |         -6.03 |        -16.38 |         0 |
