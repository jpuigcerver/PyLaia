import io

import pytest
import torch

from laia.utils import kaldi


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize(
    "device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
)
def test_write_binary_matrix(dtype, device):
    f = io.BytesIO()
    x = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=dtype, device=device)
    kaldi.write_binary_matrix(f, x)
    # Expected code
    header = b"FM " if dtype == torch.float32 else b"DM "
    # Expected rows
    header = header + b"\x04" + b"\x02\x00\x00\x00"
    # Expected columns
    header = header + b"\x04" + b"\x04\x00\x00\x00"
    assert f.getvalue() == header + x.cpu().numpy().tobytes()


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize(
    "device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
)
def test_write_text_lattice(dtype, device):
    f = io.StringIO()
    x = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=dtype, device=device)
    kaldi.write_text_lattice(f, x, digits=3)
    expected_value = (
        "0\t1\t1\t1\t0,1.0\n"
        "0\t1\t2\t2\t0,2.0\n"
        "0\t1\t3\t3\t0,3.0\n"
        "0\t1\t4\t4\t0,4.0\n"
        "1\t2\t1\t1\t0,5.0\n"
        "1\t2\t2\t2\t0,6.0\n"
        "1\t2\t3\t3\t0,7.0\n"
        "1\t2\t4\t4\t0,8.0\n"
        "2\t0,0\n\n"
    )
    assert f.getvalue() == expected_value


@pytest.mark.parametrize(
    "device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
)
def test_archive_matrix_writer(tmpdir, device):
    x1 = torch.rand(7, 9, dtype=torch.float, device=device)
    x2 = torch.rand(8, 8, dtype=torch.double, device=device)
    f = tmpdir / "matrix"
    writer = kaldi.ArchiveMatrixWriter(f)
    writer.write("key1", x1)
    writer.write("longerkey", x2)
    # Test written data
    binary_data = f.read_binary()
    expected_size1 = 4 + 3 + 3 + 5 + 5 + x1.numel() * x1.element_size()
    expected_size2 = 9 + 3 + 3 + 5 + 5 + x2.numel() * x2.element_size()
    assert len(binary_data) == expected_size1 + expected_size2
    assert binary_data[:7] == b"key1 \x00B"
    assert binary_data[expected_size1 : (expected_size1 + 12)] == b"longerkey \x00B"


@pytest.mark.parametrize(
    "device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
)
def test_archive_matrix_writer_write_iterable(tmpdir, device):
    x1 = torch.rand(7, 9, dtype=torch.float, device=device)
    x2 = torch.rand(8, 8, dtype=torch.double, device=device)

    f1 = tmpdir / "matrix1"
    writer1 = kaldi.ArchiveMatrixWriter(f1)
    writer1.write("key1", x1)
    writer1.write("key2", x2)

    f2 = tmpdir / "matrix2"
    writer2 = kaldi.ArchiveMatrixWriter(f2)
    writer2.write_iterable(zip(["key1", "key2"], [x1, x2]))

    assert f1.read_binary() == f2.read_binary()


@pytest.mark.parametrize(
    "device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
)
def test_archive_lattice_writer_write_iterable(tmpdir, device):
    f = tmpdir / "lattice"
    writer = kaldi.ArchiveLatticeWriter(f, digits=3, negate=True)
    x1 = torch.tensor([[1, 2], [5, 6]], dtype=torch.float, device=device).neg()
    x2 = torch.tensor([[1], [5]], dtype=torch.double, device=device)
    writer.write_iterable([("key1", x1), ("key2", x2)])
    assert f.read_text("utf-8") == (
        "key1\n"
        "0\t1\t1\t1\t0,1.0\n"
        "0\t1\t2\t2\t0,2.0\n"
        "1\t2\t1\t1\t0,5.0\n"
        "1\t2\t2\t2\t0,6.0\n"
        "2\t0,0\n\n"
        "key2\n"
        "0\t1\t1\t1\t0,-1.0\n"
        "1\t2\t1\t1\t0,-5.0\n"
        "2\t0,0\n\n"
    )
