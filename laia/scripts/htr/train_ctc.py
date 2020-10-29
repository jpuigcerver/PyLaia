#!/usr/bin/env python3
import argparse
from distutils.version import StrictVersion
from os.path import join

import pytorch_lightning as pl
import torch

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

    exp_dirpath = join(args.train_path, args.experiment_dirname)
    loader = ModelLoader(args.train_path, filename=args.model_filename, device="cpu")
    # maybe load a checkpoint
    checkpoint = (
        loader.prepare_checkpoint(args.checkpoint, exp_dirpath, args.monitor)
        if args.resume_training
        else None
    )
    # load the non-pytorch_lightning model
    model = loader.load()
    if model is None:
        log.error('Could not find the model. Have you run "pylaia-htr-create-model"?')
        exit(1)

    # prepare the symbols
    syms = SymbolsTable(args.syms)
    for d in args.delimiters:
        if d not in syms:
            log.error('The delimiter "{}" is not available in the symbols file', d)
            exit(1)

    # prepare the engine
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

    # prepare the data
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

    # prepare the training callbacks
    # TODO: save on lowest_va_wer and every k epochs https://github.com/PyTorchLightning/pytorch-lightning/issues/2908
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=exp_dirpath,
        filename="{epoch}-lowest_" + args.monitor,
        monitor=args.monitor,
        verbose=True,
        save_top_k=args.checkpoint_k,
        mode="min",
        save_last=True,
    )
    checkpoint_callback.CHECKPOINT_NAME_LAST = "{epoch}-last"
    early_stopping_callback = pl.callbacks.EarlyStopping(
        monitor=args.monitor,
        patience=args.early_stopping_patience,
        verbose=True,
        mode="min",
        strict=False,  # training_step may return None
    )
    callbacks = [
        ProgressBar(refresh_rate=args.lightning.progress_bar_refresh_rate),
        early_stopping_callback,
        checkpoint_callback,
    ]
    if args.gpu_stats:
        callbacks.append(ProgressBarGPUStats())
    if args.scheduler:
        callbacks.append(LearningRate(logging_interval="epoch"))

    # prepare the trainer
    trainer = pl.Trainer(
        default_root_dir=args.train_path,
        checkpoint_callback=checkpoint_callback,
        resume_from_checkpoint=checkpoint,
        callbacks=callbacks,
        logger=EpochCSVLogger(exp_dirpath),
        terminate_on_nan=False,
        **vars(args.lightning),
    )

    # train!
    trainer.fit(engine_module, datamodule=data_module)

    # training is over
    if early_stopping_callback.stopped_epoch:
        log.info(
            "Early stopping triggered after epoch"
            f" {early_stopping_callback.stopped_epoch + 1} (waited for"
            f" {early_stopping_callback.wait_count} epochs). The best score was"
            f" {early_stopping_callback.best_score}"
        )
    log.info(
        f"Model has been trained for {trainer.current_epoch + 1} epochs"
        f" ({trainer.global_step + 1} steps)"
    )
    log.info(
        f"Best {checkpoint_callback.monitor}={checkpoint_callback.best_model_score} "
        f"obtained with model={checkpoint_callback.best_model_path}"
    )


def get_args() -> argparse.Namespace:
    parser = LaiaParser().add_defaults(
        "batch_size",
        "learning_rate",
        "momentum",
        "weight_l2_penalty",
        "nesterov",
        "seed",
        "train_path",
        "monitor",
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
        "--resume_training",
        type=pl.utilities.parsing.str_to_bool,
        nargs="?",
        const=True,
        default=False,
        help="Whether to resume training with a checkpoint. See --checkpoint",
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
        choices=[f"va_{m}" for m in ("loss", "cer", "wer")],
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
    pl_group = add_lightning_args(
        pl_group,
        blocklist=[
            "checkpoint_callback",
            "default_root_dir",
            "resume_from_checkpoint",
            "log_gpu_memory",
            "logger",
            "terminate_on_nan",
            # TODO: support these 2
            "auto_scale_batch_size",
            "auto_lr_find",
        ],
    )

    args = parser.parse_args()

    # Move lightning default arguments to their own namespace
    args = group_to_namespace(args, pl_group, "lightning")
    # Hard-code some args
    args.lightning.weights_summary = "full"

    if (
        StrictVersion(torch.__version__) < StrictVersion("1.7.0")
        and args.lightning.precision != 32
    ):
        log.error(
            "AMP requires torch>=1.7.0. Additionally, only "
            "fixed height models are currently supported"
        )
        exit(1)

    return args


def main():
    run(get_args())


if __name__ == "__main__":
    main()
