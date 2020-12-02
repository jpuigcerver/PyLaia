from math import sqrt
from pathlib import Path

import laia.common.logging as log
from benchmarks.common import setup
from laia.common.arguments import (
    CommonArgs,
    DataArgs,
    OptimizerArgs,
    TrainArgs,
    TrainerArgs,
)
from laia.scripts.htr.train_ctc import run as train

log.config(to_stderr_level=log.Level.INFO, filepath="half.log", overwrite=True)

train_path = Path(__file__).parent / "half"
train_path.mkdir(exist_ok=True)

seed, data_module, syms = setup(train_path, fixed_input_height=28)

epochs = 40
batch_size = 320 * 2  # 16 bit allows double the batch size
learning_rate = 1e-3

train(
    syms,
    [str(data_module.root / p) for p in ("tr", "va")],
    *[str(data_module.root / f"{p}.gt") for p in ("tr", "va")],
    common=CommonArgs(
        train_path=train_path,
        seed=seed,
        experiment_dirname="",
    ),
    data=DataArgs(batch_size=batch_size),
    optimizer=OptimizerArgs(learning_rate=learning_rate),
    train=TrainArgs(
        # disable checkpointing
        checkpoint_k=0,
        # disable early stopping
        early_stopping_patience=epochs,
        gpu_stats=True,
    ),
    trainer=TrainerArgs(
        max_epochs=epochs,
        weights_summary=None,
        gpus=1,
        precision=16,
        # training is still not deterministic on GPU
        deterministic=True,
    ),
)
