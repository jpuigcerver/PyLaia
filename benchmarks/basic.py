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

log.config(to_stderr_level=log.Level.INFO, filepath="basic.log", overwrite=True)

train_path = Path(__file__).parent / "basic"
train_path.mkdir(exist_ok=True)

seed, data_module, syms = setup(train_path)

epochs = 40
batch_size = 320
# note: batch size 128 is able to achieve similar performance with only 15 epochs,
# however, since we are more interested in maximizing the available memory, we use
# this larger batch size.
# 1e-3 was the best learning rate with batch size 128
k = batch_size / 128
learning_rate = 1e-3 * sqrt(k)

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
        # training is still not deterministic on GPU
        deterministic=True,
    ),
)
