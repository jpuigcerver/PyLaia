# Benchmarks

This is a minimal collection of training benchmarks used to evaluate PyLaia's performance.

You can think of these as an extended suite of training tests which require a GPU, thus cannot be run in CI.

### Data

All the tests use a synthetic dataset we call "MNIST-lines", where MNIST digits are randomly selected to form text-line images, with spaces randomly added.

For larger experiments using real datasets, please have a look at the PyLaia examples [repository](https://github.com/carmocca/PyLaia-examples).

### Running

The following are available. Note that all of them require that CUDA is available:

- `basic.py`: Run Laia's CRNN model for a fixed number of epochs.
- `distributed.py`: On 2 GPUs.
- `half.py`: Using AMP's 16bit precision.
