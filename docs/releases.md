# Releases

## 1.1.0

Released on **22 December 2023** &bull; View on [Gitlab](https://gitlab.teklia.com/atr/pylaia/-/releases/1.1.0)

### Breaking changes

- Official support for Python3.8 has been dropped. This doesn't mean that the current code doesn't run on python3.8, we simply do not test that compatibility anymore. This decision was made since active support of python 3.8 has stopped for a while now and many libraries in the ML world have stopped supporting it as well.

### Feature

- A Docker image with the needed code to use this library is now built on every tag.
- The coverage of our tests suite is displayed again as a GitLab badge on the repository as well as in the README.md file.

### Documentation

- Many sections were added to the documentation:

    - for the [pylaia-htr-create-model](https://atr.pages.teklia.com/pylaia/usage/initialization/) command,
    - for [dataset formatting](https://atr.pages.teklia.com/pylaia/usage/datasets/),
    - for the [pylaia-htr-train-ctc](https://atr.pages.teklia.com/pylaia/usage/training/) command and [fine-tuning](https://atr.pages.teklia.com/pylaia/usage/training/#resume-training-from-a-checkpoint),
    - for the [pylaia-htr-decode-ctc](https://atr.pages.teklia.com/pylaia/usage/prediction/) command,
    - for the [pylaia-htr-netout](https://atr.pages.teklia.com/pylaia/usage/netout/) command,
    - to [train](https://atr.pages.teklia.com/pylaia/usage/language_models/) [KenLM](https://kheafield.com/code/kenlm/) language models,
    - the full Python code reference.

- A contribution guide and a code of conduct were added for new contributors.

### Dependencies

- Bumped [pytorch-lightning](https://pypi.org/project/pytorch-lightning/) to version `1.3.0`
- Some dependencies were pinned to a version to avoid breakage:

    - [natsort](https://pypi.org/project/natsort/) was pinned to version `8.4.0`,
    - [textdistance](https://pypi.org/project/textdistance/) was pinned to version `4.6.0`,
    - [scipy](https://pypi.org/project/scipy/) was pinned to version `1.11.3`,
    - [matplotlib](https://pypi.org/project/matplotlib/) was pinned to version `3.8.2`,
    - [numpy](https://pypi.org/project/numpy/) direct dependency was removed since it's installed through `scipy` and `matplotlib`.

- PyLaia dropped support for python 3.8 so the [dataclasses](https://pypi.org/project/dataclasses/) dependency was dropped.

### Misc

- The `torch.testing.assert_allclose` has been replaced by `torch.testing.assert_close` since it became deprecated in [PyTorch 1.12.0](https://github.com/pytorch/pytorch/issues/61844).


## 1.0.7

Released on **18 October 2023** &bull; View on [Gitlab](https://gitlab.teklia.com/atr/pylaia/-/releases/1.0.7)

### Feature
- When using a language model, a confidence score is now returned based on the log-likelyhood of the hypothesis.

### Documentation
A public documentation is now available on <https://atr.pages.teklia.com/pylaia/>. It's still under construction but next releases will add more and more content.

### Dependencies
- Bumped [pytorch-lightning](https://pypi.org/project/pytorch-lightning/) to version `1.1.7`
- Bumped GitHub action [codecov/codecov-action](https://github.com/codecov/codecov-action) to version `3`
- Bumped GitHub action [actions/setup-python](https://github.com/actions/setup-python) to version `4`
- Bumped GitHub action [actions/checkout](https://github.com/actions/checkout) to version `4`

### Development
- Releases are now built more easily through a Makefile.
- The documentation is also redeployed after each push on `master` branch.
- Fixed a test that behaved differently locally and during CI.

## 1.0.6

Released on **12 September 2023** &bull; View on [Github](https://github.com/jpuigcerver/PyLaia/releases/tag/1.0.6)

### Feature
- During training, too small images are now padded to be able to pass the multiple convolution layers.

### Documentation
- Fixed typos.

### Dependencies
- Replaced [deprecated Pillow resampling method](https://pillow.readthedocs.io/en/stable/releasenotes/2.7.0.html#antialias-renamed-to-lanczos) `Image.ANTIALIAS` to `Image.Resample.Lanczos`.

### Development
- Pre-commit hooks were updated.

## 1.0.5

Released on **29 March 2023** &bull; View on [Github](https://github.com/jpuigcerver/PyLaia/releases/tag/1.0.5)

### Dependencies
- Requires `torch` version `1.13.0` or `1.13.1`.
- Requires `torchvision` version `0.14.0` or `0.14.1` (depending on `torch` version).

## 1.0.4

Released on **4 January 2023** &bull; View on [Github](https://github.com/jpuigcerver/PyLaia/releases/tag/1.0.4)

### Dependencies
- Requires `torch` version `1.13.0`.

## 1.0.3

Released on **12 December 2022** &bull; View on [Github](https://github.com/jpuigcerver/PyLaia/releases/tag/1.0.3)

### Feature
- Now able to decode using a trained Language model through beam search decoding.
- Exposes [torch Dataloaders's num_workers](https://pytorch.org/docs/stable/data.html#multi-process-data-loading) parameter on the Python training function to limit resource usage when needed.

### Dependencies
- Added dependency to `torchaudio` version `0.13.0`.

### Development
- Package version is now tracked through the `VERSION` file.

## 1.0.2

Released on **7 December 2022** &bull; View on [Github](https://github.com/jpuigcerver/PyLaia/releases/tag/1.0.2)

### Dependencies
- Pinned dependency to `pytorch-lightning` to version `1.1.0`.

## 1.0.1

Released on **7 December 2022** &bull; View on [Github](https://github.com/jpuigcerver/PyLaia/releases/tag/1.0.1)

## 1.0.0

Released on **2 December 2020** &bull; View on [Github](https://github.com/jpuigcerver/PyLaia/releases/tag/1.0.0)

### Added

- Support distributed training
- Scripts can now be configured using yaml configuration files
- Add support for the SGD and Adam optimizers
- Support color images
- Log the installed version of each module when scripts are called from shell
- Add char/word segmentation to the decode script
- Add several badges to the README
- Support using a `ReduceLROnPlateau` scheduler during training
- A CSV file (metrics.csv) is now created with the results obtained during training
- Add CONTRIBUTING file
- Training now can include GPU stats in the progress bar
- Add isort to pre-commit to keep consistent imports throughout the codebase
- Users can run the PyLaia scripts using Python now
- Support half-precision training for fixed height models.
- Add script to visualize the segmentation output
- Use Codecov to produce test coverage reports
- Code is now analyzed using CodeFactor

### Changed

- Make Python 3.6 the minimum supported version
- Make PyTorch 1.4.0 the minimum supported version
- Remove `ImageToTensor` in favor of vision transform `ToImageTensor`
- Remove all of the internal logic (`engine`, `actions`, `hooks`, etc) in favor of pytorch-lightning's constructs
- Change Travis CI for GitHub actions
- Greatly improve the progress bar. It is used now in all scripts
- The entire shell API has changed for the better (thanks to jsonargparse). Arguments are now separated into groups and help messages are clearer.
- Drastically improve our test suite, we now have a 91% coverage

### Removed

- Remove egs directory. These live now at https://github.com/carmocca/PyLaia-examples
- Remove Baidu's CTC loss in favor of PyTorch's
- Remove PHOC code. Please open an issue if you were using it
- Remove Dortmund code. Please open an issue if you were using it
- Remove CTCLatticeGenerator. Please open an issue if you were using it
- We no longer support saving checkpoints for more than one metric. Will be added back in a future version

### Fixed

- Fix WER calculation when long delimiters are used
- Exit training if a delimiter is not present in the vocabulary
- Hundreds of other minor fixes and refactors to improve the code quality!
