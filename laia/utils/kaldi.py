from __future__ import absolute_import

import io
import numpy as np
import sys
import torch

from typing import Iterable, Tuple, Union


def write_binary_matrix(f, mat):
    """Write a matrix in Kaldi's binary format into a file-like object."""
    # type: (io.IOBase, Union[torch.Tensor, np.ndarray]) -> None
    if isinstance(mat, torch.Tensor):
        assert mat.dim() == 2, "Input tensor must have 2 dimensions"
        # TODO(jpuigcerver): Avoid conversion to numpy.
        mat = mat.detach().numpy()
    elif isinstance(mat, np.ndarray):
        assert mat.ndim == 2, "Input array must have 2 dimensions"
    else:
        raise ValueError("Matrix must be a torch.Tensor or numpy.ndarray")

    if mat.dtype == np.float32:
        dtype = "FM ".encode("ascii")
    elif mat.dtype == np.float64:
        dtype = "DM ".encode("ascii")
    else:
        raise ValueError("Matrix dtype is not supported %r" % repr(mat.dtype))

    rows, cols = mat.shape
    rows = rows.to_bytes(length=4, byteorder=sys.byteorder)
    cols = cols.to_bytes(length=4, byteorder=sys.byteorder)
    f.write(dtype + b"\x04" + rows + b"\x04" + cols)
    f.write(mat.tobytes())


def write_text_lattice(f, mat, digits=8):
    """Write a matrix as a CTC lattice in Kaldi's text format into a file-like object."""
    # type: (io.IOBase, Union[torch.Tensor, np.ndarray]) -> None
    if isinstance(mat, torch.Tensor):
        assert mat.dim() == 2, "Input tensor must have 2 dimensions"
        mat = mat.detach().numpy()
    elif isinstance(mat, np.ndarray):
        assert mat.ndim == 2, "Input array must have 2 dimensions"
    else:
        raise ValueError("Matrix must be a torch.Tensor or numpy.ndarray")

    rows, cols = mat.shape
    f.write(
        "\n".join(
            "{:d}\t{:d}\t{:d}\t{:d}\t0,{:.{digits}}".format(
                row, row + 1, col + 1, col + 1, mat[row, col], digits=digits
            )
            for row in range(rows)
            for col in range(cols)
        )
        + "\n"
        + "{:d}\t0,0\n\n".format(rows)
    )


class ArchiveMatrixWriter(object):
    """Class to write a Kaldi's archive file containing binary matrixes."""

    def __init__(self, f):
        # type: (Union[io.IOBase, str]) -> None
        if isinstance(f, str):
            self._file = io.open(f, "wb")
            self._owns_file = True
        else:
            self._file = f
            self._owns_file = False

    def __del__(self):
        if self._owns_file:
            self._file.close()

    def write(self, key, matrix):
        """Write a matrix with the given key to the archive.

        Args:
          key: the key for the given matrix.
          matrix: the matrix to write.
        """
        # type: (str, Union[torch.Tensor, np.ndarray]) -> None
        if not isinstance(key, str):
            raise ValueError("Key %r is not a string" % repr(key))
        self._file.write(("%s " % key).encode("utf-8"))
        self._file.write(b"\x00B")
        write_binary_matrix(self._file, matrix)

    def write_iterable(self, iterable):
        """Write a collection of matrixes to the archive from an iterable.

        Args:
          iterable: iterable containing pairs of (key, matrix).
        """
        # type: Iterable[Tuple[str, Union[torch.Tensor, np.ndarray]]]) -> None
        for key, mat in iterable:
            self.write(key, mat)


class ArchiveLatticeWriter(object):
    """Class to write a Kaldi's archive file containing text lattices."""

    def __init__(self, f, digits=8, negate=False):
        # type: (Union[io.IOBase, str], int) -> None
        if isinstance(f, str):
            self._file = io.open(f, "w")
            self._owns_file = True
        else:
            self._file = f
            self._owns_file = False

        self._digits = digits
        self._negate = negate

    def __del__(self):
        if self._owns_file:
            self._file.close()

    def write(self, key, matrix):
        """Write a matrix with the given key to the archive.

        Args:
          key: the key for the given matrix.
          matrix: the matrix to write.
        """
        # type: (str, Union[torch.Tensor, np.ndarray]) -> None
        if not isinstance(key, str):
            raise ValueError("Key %r is not a string" % repr(key))
        self._file.write("{}\n".format(key))
        if self._negate:
            matrix = -matrix
        write_text_lattice(self._file, matrix, digits=self._digits)

    def write_iterable(self, iterable):
        """Write a collection of matrixes to the archive from an iterable.

        Args:
          iterable: iterable containing pairs of (key, matrix).
        """
        # type: Iterable[Tuple[str, Union[torch.Tensor, np.ndarray]]]) -> None
        for key, mat in iterable:
            self.write(key, mat)
