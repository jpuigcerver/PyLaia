import sys
from typing import BinaryIO, Iterable, TextIO, Tuple, Union

import numpy as np
import torch


def prepare_mat(mat: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    if isinstance(mat, torch.Tensor):
        assert mat.dim() == 2, "Input tensor must have 2 dimensions"
        # TODO(jpuigcerver): Avoid conversion to numpy.
        mat = mat.detach().cpu().numpy()
    elif isinstance(mat, np.ndarray):
        assert mat.ndim == 2, "Input array must have 2 dimensions"
    else:
        raise ValueError("Matrix must be a torch.Tensor or numpy.ndarray")
    return mat


def write_binary_matrix(f: BinaryIO, mat: Union[torch.Tensor, np.ndarray]) -> None:
    """Write a matrix in Kaldi's binary format into a file-like object."""
    mat = prepare_mat(mat)
    if mat.dtype == np.float32:
        dtype = "FM ".encode("ascii")
    elif mat.dtype == np.float64:
        dtype = "DM ".encode("ascii")
    else:
        raise ValueError(f"Matrix dtype is not supported {mat.dtype}")

    rows, cols = mat.shape
    rows = rows.to_bytes(length=4, byteorder=sys.byteorder)
    cols = cols.to_bytes(length=4, byteorder=sys.byteorder)
    f.write(dtype + b"\x04" + rows + b"\x04" + cols)
    f.write(mat.tobytes())


def write_text_lattice(
    f: TextIO, mat: Union[torch.Tensor, np.ndarray], digits: int = 8
) -> None:
    """Write a matrix as a CTC lattice in Kaldi's text format into a file-like object."""
    mat = prepare_mat(mat)
    rows, cols = mat.shape
    f.write(
        "\n".join(
            f"{row:d}\t{row + 1:d}\t{col + 1:d}\t{col + 1:d}\t0,{mat[row, col]:.{digits}}"
            for row in range(rows)
            for col in range(cols)
        )
        + "\n"
        + f"{rows:d}\t0,0\n\n"
    )


class ArchiveMatrixWriter:
    """Class to write a Kaldi's archive file containing binary matrices."""

    def __init__(self, f: Union[BinaryIO, str]) -> None:
        if isinstance(f, str):
            self._file = open(f, "wb")
            self._owns_file = True
        else:
            self._file = f
            self._owns_file = False

    def __del__(self):
        if self._owns_file:
            self._file.close()

    def write(self, key: str, matrix: Union[torch.Tensor, np.ndarray]) -> None:
        """Write a matrix with the given key to the archive.

        Args:
          key: the key for the given matrix.
          matrix: the matrix to write.
        """
        if not isinstance(key, str):
            raise ValueError(f"Key {key} is not a string")
        self._file.write(f"{key} ".encode("utf-8"))
        self._file.write(b"\x00B")
        write_binary_matrix(self._file, matrix)

    def write_iterable(
        self, iterable: Iterable[Tuple[str, Union[torch.Tensor, np.ndarray]]]
    ) -> None:
        """Write a collection of matrices to the archive from an iterable.

        Args:
          iterable: iterable containing pairs of (key, matrix).
        """
        for key, mat in iterable:
            self.write(key, mat)


class ArchiveLatticeWriter:
    """Class to write a Kaldi's archive file containing text lattices."""

    def __init__(
        self, f: Union[TextIO, str], digits: int = 8, negate: bool = False
    ) -> None:
        if isinstance(f, str):
            self._file = open(f, "w")
            self._owns_file = True
        else:
            self._file = f
            self._owns_file = False

        self._digits = digits
        self._negate = negate

    def __del__(self):
        if self._owns_file:
            self._file.close()

    def write(self, key: str, matrix: Union[torch.Tensor, np.ndarray]) -> None:
        """Write a matrix with the given key to the archive.

        Args:
          key: the key for the given matrix.
          matrix: the matrix to write.
        """
        if not isinstance(key, str):
            raise ValueError(f"Key {key} is not a string")
        self._file.write(f"{key}\n")
        if self._negate:
            matrix = -matrix
        write_text_lattice(self._file, matrix, digits=self._digits)

    def write_iterable(
        self, iterable: Iterable[Tuple[str, Union[torch.Tensor, np.ndarray]]]
    ) -> None:
        """Write a collection of matrices to the archive from an iterable.

        Args:
          iterable: iterable containing pairs of (key, matrix).
        """
        for key, mat in iterable:
            self.write(key, mat)
