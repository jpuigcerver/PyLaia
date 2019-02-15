# PyLaia

[![Build Status](https://travis-ci.com/jpuigcerver/PyLaia.svg?token=HF64eTvPxEUcjjUPXpgm&branch=master)](https://travis-ci.com/jpuigcerver/PyLaia)
[![Python Version](https://img.shields.io/badge/python-3.5%2C%203.6%2C%203.7-blue.svg)](https://www.python.org/)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

PyLaia is a device agnostic, PyTorch based, deep learning toolkit specialized for handwritten document analysis. It is also a successor to [Laia](https://github.com/jpuigcerver/Laia).

## Requirements

The file `[requirements.txt](https://github.com/jpuigcerver/PyLaia/blob/master/requirements.txt)`
includes all the Python packages required to install and use PyLaia.

The recipes for some datasets also need additional tools and packages.
For instance, typically ImageMagick is used to process images, or Kaldi
is employed to perform Viterbi decoding (and lattice generation) combining
the output of the neural network with a n-gram language model.

## Usage

### Training a Laia model using CTC:

Create a model using:

```bash
$ pylaia-htr-create-model --fixed_input_height="$INPUT_HEIGHT" "$CHANNELS" "$SYM_TABLE"
```

Required arguments:

- `$CHANNELS`: Number of channels of the input images.
- `$SYM_TABLE`: Path to the table file mapping symbols to their ids.
- Unless you have installed nnutils, you will need to use models that process images
  of a fixed height. Use `--fixed_input_height=$INPUT_HEIGHT` to specify the height of
  the image.

For optional arguments check `$ pylaia-htr-create-model -h`

Train the model using:

```bash
$ pylaia-htr-train-ctc "$SYM_TABLE" "$IMG_DIRS" "$TRAIN_GT" "$VALID_GT"
```

Required arguments:

- `$SYM_TABLE`: Path to the table file mapping symbols to their ids.
- `$IMG_DIRS`: Directory(s) where the images are located.
- `$TRAIN_GT`: A file containing the list of training transcripts.
- `$VALID_GT`: A file containing the list of validation transcripts.

For optional arguments check `$ pylaia-htr-train-ctc -h`

### Transcribing

```bash
$ pylaia-htr-decode-ctc "$SYM_TABLE" "$IMG_DIRS" "$IMG_LIST"
```

Required arguments:

- `$SYM_TABLE`: Path to the table file mapping symbols to their ids.
- `$IMG_DIRS`: Directory(s) where the images are located. This is ignored if `$IMG_LIST` elements are full paths.
- `$IMG_LIST`: A file containing a list of images to transcribe. They can be image ids with or without extension or full paths to the images.

For optional arguments check `$ pylaia-htr-decode-ctc -h`
