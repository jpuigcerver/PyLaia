# PyLaia

[![Build Status](https://travis-ci.com/jpuigcerver/PyLaia.svg?token=HF64eTvPxEUcjjUPXpgm&branch=master)](https://travis-ci.com/jpuigcerver/PyLaia) [![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

PyLaia is a device agnostic, PyTorch based, deep learning toolkit to transcribe handwritten text images. It is also a succesor to [Laia](https://github.com/jpuigcerver/Laia).

## Requirements

- [PyTorch 0.3.1](https://pytorch.org)
- [warp-ctc](./third_party/warp-ctc)
- [nnutils](./third_party/nnutils)

## Optional requirements (depending on your usage)

- [prob-phoc](./third_party/prob-phoc)
- [imgdistort](./third_party/imgdistort)

## Usage

### Training a Laia model using CTC:

Create a model using:

```bash
$ pylaia-htr-create-model \
    "$INPUT_HEIGHT" "$CHANNELS" "$SYM_TABLE" \
    --train_path="$TRAIN_PATH";
```

Required arguments:

- `$INPUT_HEIGHT`: Height of the input images. Set this to 0 to use a variable height model.
- `$CHANNELS`: Number of channels of the input images.
- `$SYM_TABLE`: Path to the table file mapping symbols to their ids.
- `$TRAIN_PATH`: Location where the model will be saved.

For optional arguments check `$ pylaia-htr-create-model -h`

Train the model using:

```bash
$ pylaia-htr-train-ctc \
    "$SYM_TABLE" "$IMG_DIRS" "$TRAIN_GT" "$VALID_GT" \
    --train_path="$TRAIN_PATH";
```

Required arguments:

- `$SYM_TABLE`: Path to the table file mapping symbols to their ids.
- `$IMG_DIRS`: Directory(s) where the images are located.
- `$TRAIN_GT`: A file containing the list of training transcripts.
- `$VALID_GT`: A file containing the list of validation transcripts.
- `$TRAIN_PATH`: Location where the trainer and checkpoints will be saved.

For optional arguments check `$ pylaia-htr-train-ctc -h`

### Transcribing

```bash
$ pylaia-htr-decode-ctc \
    "$SYM_TABLE" "$IMG_DIRS" "$IMG_LIST" \
    --train_path="$TRAIN_PATH";
```

Required arguments:

- `$SYM_TABLE`: Path to the table file mapping symbols to their ids.
- `$IMG_DIRS`: Directory(s) where the images are located. This is ignored if `$IMG_LIST` elements are full paths.
- `$IMG_LIST`: A file containing a list of images to transcribe. They can be image ids with or without extension or full paths to the images.
- `$TRAIN_PATH`: Location where the model and its checkpoint is located.

For optional arguments check `$ pylaia-htr-decode-ctc -h`
