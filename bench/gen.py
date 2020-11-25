import subprocess as sp
import torch as th


def run_bench(name, *args, device="cpu"):
    args = list(args)
    args += ["-d", device]
    if device == "cuda" and not th.cuda.is_available():
        return "Not available /!\\"
    return sp.check_output(["python3", "-m", f"bench.{name}"] + args).decode('utf8')


def main():
    template = f"""\
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

{run_bench('resample')}

On GPU we have:

{run_bench('resample', device='cuda')}

### FFTConv1d

We compare to `pytorch.nn.functional.conv1d`, on a input of size [32, 32, 10240],
for a convolution with 32 input channels, 64 output channels and various kernel sizes.

On CPU we have:

{run_bench('fftconv')}

On GPU we have:

{run_bench('fftconv', device='cuda')}

### LowPassFilter

We do not compare to anything, but measure the attenuation in dB of a pure tone
at `0.9 * cutoff`, at the `cutoff`, and at `1.1 * cutoff`.
Note that our implementation automatically choses to use FFTConv1d or not when appropriate.

On CPU we have:

{run_bench('lowpass')}

On GPU we have:

{run_bench('lowpass', device='cuda')}

"""
    print(template)


if __name__ == "__main__":
    main()
