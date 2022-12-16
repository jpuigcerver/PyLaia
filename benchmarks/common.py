from pytorch_lightning import seed_everything

from laia.common.arguments import CommonArgs, CreateCRNNArgs
from laia.dummies import DummyMNISTLines
from laia.scripts.htr.create_model import run as model
from laia.utils import SymbolsTable


def setup(train_path, fixed_input_height=0):
    seed = 31102020
    seed_everything(seed)

    n = 10**4
    data_module = DummyMNISTLines(tr_n=n, va_n=int(0.1 * n), samples_per_space=5)
    print("Generating data...")
    data_module.prepare_data()

    syms = str(train_path / "syms")
    syms_table = SymbolsTable()
    for k, v in data_module.syms:
        syms_table.add(k, v)
    syms_table.save(syms)

    model(
        syms,
        adaptive_pooling="avgpool-3",
        fixed_input_height=fixed_input_height,
        save_model=True,
        common=CommonArgs(train_path=train_path),
        crnn=CreateCRNNArgs(
            cnn_num_features=[16, 32, 48, 64],
            # data is random so minimal RNN layer
            # because there are no long term dependencies
            rnn_units=32,
            rnn_layers=1,
            rnn_dropout=0,
        ),
    )

    return seed, data_module, syms
