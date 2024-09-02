<div align="center">

# PyLaia

**PyLaia is a device agnostic, PyTorch based, deep learning toolkit for handwritten document analysis.**

**It is also a successor to [Laia](https://github.com/jpuigcerver/Laia).**

**Development/Maintenance is now done on [Teklia's fork](https://gitlab.teklia.com/atr/pylaia). Please head over there if you want to use the latest version of PyLaia.**

[![Build](https://img.shields.io/github/workflow/status/jpuigcerver/PyLaia/Laia%20CI?&label=Build&logo=GitHub&labelColor=1b1f23)](https://github.com/jpuigcerver/PyLaia/actions?query=workflow%3A%22Laia+CI%22)
[![Coverage](https://img.shields.io/codecov/c/github/jpuigcerver/PyLaia?&label=Coverage&logo=Codecov&logoColor=ffffff&labelColor=f01f7a)](https://codecov.io/gh/jpuigcerver/PyLaia)
[![Code quality](https://img.shields.io/codefactor/grade/github/jpuigcerver/PyLaia?&label=CodeFactor&logo=CodeFactor&labelColor=2782f7)](https://www.codefactor.io/repository/github/jpuigcerver/PyLaia)

[![Python: 3.8+](https://img.shields.io/badge/Python-3.8%2B-FFD43B.svg?&logo=Python&logoColor=white&labelColor=306998)](https://www.python.org/)
[![PyTorch: 1.13.0+](https://img.shields.io/badge/PyTorch-1.13.0%2B-8628d5.svg?&logo=PyTorch&logoColor=white&labelColor=%23ee4c2c)](https://pytorch.org/)
[![pre-commit: enabled](https://img.shields.io/badge/pre--commit-enabled-76877c?&logo=pre-commit&labelColor=1f2d23)](https://github.com/pre-commit/pre-commit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg?)](https://github.com/ambv/black)

</div>

Get started by having a look at our [Wiki](https://github.com/jpuigcerver/PyLaia/wiki)!
###### Several (mostly undocumented) examples of its use are provided at [PyLaia-examples](https://github.com/carmocca/PyLaia-examples).

## Installation

In order to install PyLaia, follow this recipe:

```bash
git clone https://github.com/jpuigcerver/PyLaia
cd PyLaia
pip install -e .
```

Please note that the CUDA version of nnutils ([`nnutils-pytorch-cuda`](https://pypi.org/project/nnutils-pytorch-cuda/)) is installed by default. If you do not have a GPU, you should install the CPU version ([`nnutils-pytorch`](https://pypi.org/project/nnutils-pytorch/)).

The following Python scripts will be installed in your system:

- [`pylaia-htr-create-model`](laia/scripts/htr/create_model.py): Create a VGG-like model with BLSTMs on top for handwriting text recognition. The script has different options to customize the model. The architecture is based on the paper ["Are Multidimensional Recurrent Layers Really Necessary for Handwritten Text Recognition?"](https://ieeexplore.ieee.org/document/8269951) (2017) by J. Puigcerver.
- [`pylaia-htr-train-ctc`](laia/scripts/htr/train_ctc.py): Train a model using the CTC algorithm and a set of text-line images and their transcripts.
- [`pylaia-htr-decode-ctc`](laia/scripts/htr/decode_ctc.py): Decode text line images using a trained model and the CTC algorithm. It can also output the char/word segmentation boundaries of the symbols recognized.
- [`pylaia-htr-netout`](laia/scripts/htr/netout.py): Dump the output of the model for a set of text-line images in order to decode using an external language model.

## Acknowledgments

Work in this toolkit was financially supported by the [Pattern Recognition and Human Language Technology (PRHLT) Research Center](https://www.prhlt.upv.es/wp/)

## BibTeX

```
@misc{puigcerver2018pylaia,
  author = {Joan Puigcerver and Carlos Mocholí},
  title = {PyLaia},
  year = {2018},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/jpuigcerver/PyLaia}},
  commit = {commit SHA}
}
```
