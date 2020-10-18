import argparse

from laia.models.htr import LaiaCRNN
from laia.scripts.htr.create_model import run
from laia.utils import SymbolsTable


def test_run_fixed_height(monkeypatch):
    monkeypatch.setattr(SymbolsTable, "__len__", lambda _: 80)
    args = argparse.Namespace(
        num_input_channels=1,
        syms=None,
        cnn_num_features=[16, 32, 48, 64, 80],
        cnn_kernel_size=[(3, 3)] * 5,
        cnn_stride=[(1, 1)] * 5,
        cnn_dilation=[(1, 1)] * 5,
        cnn_activation=["LeakyReLU"] * 5,
        cnn_poolsize=[(2, 2)] * 3 + [(0, 0)] * 2,
        cnn_dropout=[0] * 5,
        cnn_batchnorm=[False] * 5,
        rnn_units=256,
        rnn_layers=5,
        rnn_dropout=0.5,
        lin_dropout=0.5,
        rnn_type="LSTM",
        vertical_text=False,
        use_masked_conv=False,
        fixed_input_height=128,
    )
    model = run(args)
    assert isinstance(model, LaiaCRNN)
    assert sum(param.numel() for param in model.parameters()) == 9591248


def test_run_variable_height(tmpdir, monkeypatch):
    monkeypatch.setattr(SymbolsTable, "__len__", lambda _: 80)
    args = argparse.Namespace(
        num_input_channels=1,
        syms=None,
        cnn_num_features=[16, 32, 48, 64, 80],
        cnn_kernel_size=[(3, 3)] * 5,
        cnn_stride=[(1, 1)] * 5,
        cnn_dilation=[(1, 1)] * 5,
        cnn_activation=["LeakyReLU"] * 5,
        cnn_poolsize=[(2, 2)] * 3 + [(0, 0)] * 2,
        cnn_dropout=[0] * 5,
        cnn_batchnorm=[False] * 5,
        adaptive_pooling="avgpool-16",
        rnn_units=256,
        rnn_layers=5,
        rnn_dropout=0.5,
        lin_dropout=0.5,
        rnn_type="LSTM",
        vertical_text=False,
        use_masked_conv=False,
        fixed_input_height=None,
        train_path=tmpdir,
        model_filename="test_model",
    )
    model = run(args)
    assert isinstance(model, LaiaCRNN)
    assert sum(param.numel() for param in model.parameters()) == 9591248
    assert tmpdir.join("test_model").exists()
