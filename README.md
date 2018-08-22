# PyLaia

[![Build Status](https://travis-ci.com/jpuigcerver/PyLaia.svg?token=HF64eTvPxEUcjjUPXpgm&branch=master)](https://travis-ci.com/jpuigcerver/PyLaia)
[![Python Version](https://img.shields.io/badge/python-2.7%2C%203.5%2C%203.6-blue.svg)](https://www.python.org/)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

PyLaia is a device agnostic, PyTorch based, deep learning toolkit specialized for handwritten document analysis. It is also a successor to [Laia](https://github.com/jpuigcerver/Laia).

## Requirements

- [PyTorch 0.4.1](https://pytorch.org)
- [warp-ctc](./third_party/warp-ctc)

## Optional requirements (depending on your usage)

- [nnutils](./third_party/nnutils)
- [prob-phoc](./third_party/prob-phoc)

## Usage

### Training a Laia model using CTC:

Create a model using:

```bash
$ pylaia-htr-create-model "$INPUT_HEIGHT" "$CHANNELS" "$SYM_TABLE"
```

Required arguments:

- `$INPUT_HEIGHT`: Height of the input images. Set this to 0 to use a variable height model.
- `$CHANNELS`: Number of channels of the input images.
- `$SYM_TABLE`: Path to the table file mapping symbols to their ids.

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

## FAQ

#### Where can I find the last version PyTorch 0.3.1 compatible?
You can check out the repository using this [branch](https://github.com/jpuigcerver/PyLaia/tree/PyTorch-v0.3.1)
