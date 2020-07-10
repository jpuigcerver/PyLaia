# PyLaia

[![Build Status](https://travis-ci.com/jpuigcerver/PyLaia.svg?token=HF64eTvPxEUcjjUPXpgm&branch=master)](https://travis-ci.com/jpuigcerver/PyLaia)
[![Python Version](https://img.shields.io/badge/python-3.6%2B-blue.svg)](https://www.python.org/)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

PyLaia is a device agnostic, PyTorch based, deep learning toolkit specialized
for handwritten document analysis. It is also a successor to
[Laia](https://github.com/jpuigcerver/Laia).

Several examples of its use are provided at [PyLaia-examples](https://github.com/carmocca/PyLaia-examples).

## Installation

In order to install PyLaia, follow this recipe:

```bash
git clone https://github.com/jpuigcerver/PyLaia
cd PyLaia
pip install -r requirements.txt
python setup.py install
```

The following Python scripts will be installed in your system:

- **pylaia-htr-create-model**: Create a VGG-like model with BLSTMs on top for
  handwriting text recognition. The script has different options to costumize
  the model. The architecture is based on the paper ["Are Multidimensional
  Recurrent Layers Really Necessary for Handwritten Text Recognition?"](https://ieeexplore.ieee.org/document/8269951)
  (2017) by J. Puigcerver.
- **pylaia-htr-decode-ctc**: Decode text line images using a trained model and
  the CTC algorithm.
- **pylaia-htr-train-ctc**: Train a model using the CTC algorithm and a set of
  text-line images and their transcripts.
- **pylaia-htr-netout**: Dump the output of the model for a set of text-line images
  in order to decode using an external language model.

Some examples need additional tools and packages, which are not installed
with `pip install -r requirements.txt`.
For instance, typically ImageMagick is used to process images, or Kaldi
is employed to perform Viterbi decoding (and lattice generation) combining
the output of the neural network with a n-gram language model.
