#!/usr/bin/env python3
import argparse
from os.path import join

import pytorch_lightning as pl

import laia.common.logging as log
from laia import get_installed_versions
from laia.callbacks import LearningRate, ProgressBar, ProgressBarGPUStats
from laia.common.arguments import (
    LaiaParser,
    add_lightning_args,
    get_key,
    group_to_namespace,
)
from laia.common.arguments_types import NumberInClosedRange
from laia.common.loader import ModelLoader
from laia.engine import Compose, DataModule, HTREngineModule, ImageFeeder, ItemFeeder
from laia.loggers import EpochCSVLogger
from laia.utils import SymbolsTable


def run(args: argparse.Namespace):
    log.info(f"Installed: {get_installed_versions()}")

    pl.seed_everything(get_key(args, "seed"))

    checkpoint = None
    exp_dirpath = join(args.train_path, args.experiment_dirname)
    if args.checkpoint:
        checkpoint_path = join(exp_dirpath, args.checkpoint)
        checkpoint = ModelLoader.choose_by(checkpoint_path)
        if not checkpoint:
            log.error('Could not find the checkpoint "{}"', checkpoint_path)
            exit(1)
        log.info('Using checkpoint "{}"', checkpoint)

    # Load the non-pytorch_lightning model
    model = ModelLoader(args.train_path, filename=args.model_filename).load()
    if model is None:
        log.error('Could not find the model. Have you run "pylaia-htr-create-model"?')
        exit(1)

    syms = SymbolsTable(args.syms)
    for d in args.delimiters:
        if d not in syms:
            log.error('The delimiter "{}" is not available in the symbols file', d)
            exit(1)

    optimizer_kwargs = {
        "learning_rate": args.learning_rate,
        "momentum": args.momentum,
        "weight_l2_penalty": args.weight_l2_penalty,
        "nesterov": args.nesterov,
        "scheduler": args.scheduler,
        "scheduler_patience": args.scheduler_patience,
        "scheduler_factor": args.scheduler_factor,
        "scheduler_monitor": args.scheduler_monitor,
    }
    engine_module = HTREngineModule(
        model,
        args.optimizer,
        [syms[d] for d in args.delimiters],
        optimizer_kwargs=optimizer_kwargs,
        batch_input_fn=Compose([ItemFeeder("img"), ImageFeeder()]),
        batch_target_fn=ItemFeeder("txt"),
        batch_id_fn=ItemFeeder("id"),  # Used to print image ids on exception
    )

    data_module = DataModule(
        img_dirs=args.img_dirs,
        color_mode=args.color_mode,
        batch_size=args.batch_size,
        tr_txt_table=args.tr_txt_table.name,
        va_txt_table=args.va_txt_table.name,
        tr_shuffle=not bool(args.lightning.limit_train_batches),
        tr_distortions=args.use_distortions,
        syms=syms,
        stage="fit",
    )

    early_stopping_callback = pl.callbacks.EarlyStopping(
        monitor=args.monitor,
        patience=args.early_stopping_patience,
        verbose=True,
        mode="min",
        strict=False,  # training_step may return None
    )
    # TODO: save on lowest_va_wer and every k epochs https://github.com/PyTorchLightning/pytorch-lightning/issues/2908
    pl.callbacks.ModelCheckpoint.CHECKPOINT_NAME_LAST = "{epoch}-last"
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=exp_dirpath,
        filename="{epoch}-lowest_" + args.monitor,
        monitor=args.monitor,
        verbose=True,
        save_top_k=args.checkpoint_k,
        mode="min",
        save_last=True,
    )
    callbacks = [ProgressBar(refresh_rate=args.lightning.progress_bar_refresh_rate)]
    if args.gpu_stats:
        callbacks.append(ProgressBarGPUStats())
    if args.scheduler:
        callbacks.append(LearningRate(logging_interval="epoch"))

    trainer = pl.Trainer(
        default_root_dir=args.train_path,
        early_stop_callback=early_stopping_callback,
        checkpoint_callback=checkpoint_callback,
        resume_from_checkpoint=checkpoint,
        callbacks=callbacks,
        logger=EpochCSVLogger(exp_dirpath),
        **vars(args.lightning),
    )
    trainer.fit(engine_module, datamodule=data_module)

    log.info(
        f'Best {args.monitor}="{checkpoint_callback.best_model_score}" '
        f'obtained with model="{checkpoint_callback.best_model_path}"'
    )


def get_args() -> argparse.Namespace:
    metrics = [f"va_{x}" for x in ("loss", "cer", "wer")]
    parser = LaiaParser().add_defaults(
        "batch_size",
        "learning_rate",
        "momentum",
        "weight_l2_penalty",
        "nesterov",
        "seed",
        "train_path",
        "checkpoint",
        "checkpoint_k",
        "model_filename",
        "experiment_dirname",
        "color_mode",
        "use_distortions",
    )
    parser.add_argument(
        "syms",
        type=argparse.FileType("r"),
        help="Symbols table mapping from strings to integers",
    ).add_argument(
        "img_dirs", type=str, nargs="+", help="Directory containing word images"
    ).add_argument(
        "tr_txt_table",
        type=argparse.FileType("r"),
        help="Character transcriptions of each training image",
    ).add_argument(
        "va_txt_table",
        type=argparse.FileType("r"),
        help="Character transcriptions of each validation image",
    ).add_argument(
        "--delimiters",
        type=str,
        nargs="+",
        default=["<space>"],
        help="Sequence of characters representing the word delimiters",
    ).add_argument(
        "--early_stopping_patience",
        type=NumberInClosedRange(int, vmin=0),
        default=20,
        help=(
            "Number of validation epochs with no improvement "
            "after which training will be stopped"
        ),
    ).add_argument(
        "--optimizer",
        type=str,
        default="RMSProp",
        choices=["SGD", "RMSProp", "Adam"],
        help="Optimization algorithm",
    ).add_argument(
        "--monitor",
        type=str,
        default="va_cer",
        choices=metrics,
        help="Metric to monitor for early stopping and checkpointing",
    ).add_argument(
        "--scheduler",
        type=pl.utilities.parsing.str_to_bool,
        nargs="?",
        const=True,
        default=False,
        help="Whether to use an on-plateau learning rate scheduler",
    ).add_argument(
        "--scheduler_patience",
        type=NumberInClosedRange(type=int, vmin=0),
        default=5,
        help=(
            "Number of epochs with no improvement "
            "after which learning rate will be reduced"
        ),
    ).add_argument(
        "--scheduler_monitor",
        type=str,
        default="va_loss",
        choices=metrics,
        help="Metric for the scheduler to monitor",
    ).add_argument(
        "--scheduler_factor",
        type=float,
        default=0.1,
        help="Factor by which the learning rate will be reduced",
    ).add_argument(
        "--gpu_stats",
        type=pl.utilities.parsing.str_to_bool,
        nargs="?",
        const=True,
        default=False,
        help="Whether to include GPU stats in the training progress bar",
    )

    # Add lightning default arguments to a group
    pl_group = parser.parser.add_argument_group(title="pytorch-lightning arguments")
    pl_group = add_lightning_args(pl_group)

    args = parser.parse_args()

    # Move lightning default arguments to their own namespace
    args = group_to_namespace(args, pl_group, "lightning")
    # Hard-code some args
    args.lightning.weights_summary = "full"
    # Delete some which will be set manually
    for a in (
        "checkpoint_callback",
        "default_root_dir",
        "resume_from_checkpoint",
        "log_gpu_memory",
        "logger",
    ):
        delattr(args.lightning, a)

    return args


def main():
    run(get_args())


if __name__ == "__main__":
    main()
