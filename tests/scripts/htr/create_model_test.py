from laia.common.arguments import CommonArgs, CreateCRNNArgs
from laia.models.htr import LaiaCRNN
from laia.scripts.htr.create_model import run
from laia.utils import SymbolsTable


def test_run_fixed_height(monkeypatch):
    monkeypatch.setattr(SymbolsTable, "__len__", lambda _: 80)
    model = run(
        syms=None,
        fixed_input_height=128,
        save_model=False,
        crnn=CreateCRNNArgs(
            cnn_num_features=[16, 32, 48, 64, 80],
            cnn_kernel_size=[3] * 5,
            cnn_stride=[1] * 5,
            cnn_dilation=[1] * 5,
            cnn_activation=["LeakyReLU"] * 5,
            cnn_poolsize=[2] * 3 + [0] * 2,
            cnn_dropout=[0] * 5,
            cnn_batchnorm=[False] * 5,
            rnn_layers=5,
        ),
    )
    assert isinstance(model, LaiaCRNN)
    assert sum(param.numel() for param in model.parameters()) == 9591248


def test_run_variable_height(tmpdir, monkeypatch):
    monkeypatch.setattr(SymbolsTable, "__len__", lambda _: 80)
    model = run(
        syms=None,
        save_model=True,
        common=CommonArgs(train_path=tmpdir, model_filename="test_model"),
        crnn=CreateCRNNArgs(
            cnn_num_features=[16, 32, 48, 64, 80],
            cnn_kernel_size=[3] * 5,
            cnn_stride=[1] * 5,
            cnn_dilation=[1] * 5,
            cnn_activation=["LeakyReLU"] * 5,
            cnn_poolsize=[2] * 3 + [0] * 2,
            cnn_dropout=[0] * 5,
            cnn_batchnorm=[False] * 5,
            rnn_layers=5,
        ),
    )
    assert isinstance(model, LaiaCRNN)
    assert sum(param.numel() for param in model.parameters()) == 9591248
    assert tmpdir.join("test_model").exists()
