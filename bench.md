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
|      2 |      1 |         310 |         1639 |      1.3% |
|      1 |      2 |         380 |         2646 |      1.8% |
|      4 |      5 |          94 |         1704 |      1.8% |
|     10 |     11 |          53 |         1479 |      1.8% |
|  44100 |  16000 |          70 |         2552 |      0.9% |
|  20001 |  30001 |       23788 |         2546 |      1.8% |


On GPU we have:

| Old sr | New sr | Julius (ms) |
|--------|--------|-------------|
|      2 |      1 |          44 |
|      1 |      2 |           8 |
|      4 |      5 |           5 |
|     10 |     11 |           6 |
|  44100 |  16000 |          23 |
|  20001 |  30001 |       15186 |


### FFTConv1d

We compare to `pytorch.nn.functional.conv1d`, on a input of size [32, 32, 10240],
for a convolution with 32 input channels, 64 output channels and various kernel sizes.

On CPU we have:

| Kernel size | FFT (ms) | No FFT (ms) |   Delta |
|-------------|----------|-------------|---------|
|           8 |      506 |         204 | 4.0e-06 |
|          32 |      410 |         409 | 1.4e-05 |
|          64 |      400 |         847 | 2.7e-05 |
|         128 |      356 |        1854 | 5.3e-05 |
|         256 |      350 |        3315 | 1.0e-04 |
|        1024 |      683 |       17779 | 4.1e-04 |
|        2048 |     1031 |       28133 | 8.2e-04 |


On GPU we have:

| Kernel size | FFT (ms) | No FFT (ms) |   Delta |
|-------------|----------|-------------|---------|
|           8 |       15 |           4 | 4.4e-06 |
|          32 |       12 |           5 | 1.5e-05 |
|          64 |       10 |          11 | 3.1e-05 |
|         128 |       10 |          22 | 5.4e-05 |
|         256 |        9 |          44 | 1.1e-04 |
|        1024 |       14 |         330 | 4.2e-04 |
|        2048 |       13 |         561 | 8.3e-04 |


### LowPassFilter

We do not compare to anything, but measure the attenuation in dB of a pure tone
at `0.9 * cutoff`, at the `cutoff`, and at `1.1 * cutoff`.
Note that our implementation automatically choses to use FFTConv1d or not when appropriate.

On CPU we have:

| Freq. | Attn. 0.9 (dB) | Attn 1.0 (dB) | Attn 1.1 (dB) | Time (ms) |
|-------|----------------|---------------|---------------|-----------|
| 0.005 |          -1.41 |         -6.02 |        -16.41 |         9 |
|  0.01 |          -1.41 |         -6.02 |        -16.46 |         7 |
|   0.1 |          -1.41 |         -6.02 |        -16.48 |        11 |
|   0.2 |          -1.41 |         -6.02 |        -16.48 |         5 |
|   0.4 |          -1.41 |         -6.03 |        -16.38 |         6 |


On GPU we have:

| Freq. | Attn. 0.9 (dB) | Attn 1.0 (dB) | Attn 1.1 (dB) | Time (ms) |
|-------|----------------|---------------|---------------|-----------|
| 0.005 |          -1.41 |         -6.02 |        -16.41 |         1 |
|  0.01 |          -1.41 |         -6.02 |        -16.46 |         1 |
|   0.1 |          -1.41 |         -6.02 |        -16.48 |         1 |
|   0.2 |          -1.41 |         -6.02 |        -16.48 |         0 |
|   0.4 |          -1.41 |         -6.03 |        -16.38 |         0 |


