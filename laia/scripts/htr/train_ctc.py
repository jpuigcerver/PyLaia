#!/usr/bin/env python3
from typing import Any, Dict, List, Optional

import jsonargparse
import pytorch_lightning as pl
import torch

import laia.common.logging as log
from laia.callbacks import LearningRate, ProgressBar, ProgressBarGPUStats
from laia.common.arguments import (
    CommonArgs,
    DataArgs,
    OptimizerArgs,
    SchedulerArgs,
    TrainArgs,
    TrainerArgs,
)
from laia.common.loader import ModelLoader
from laia.engine import Compose, DataModule, HTREngineModule, ImageFeeder, ItemFeeder
from laia.loggers import EpochCSVLogger
from laia.scripts.htr import common_main
from laia.utils import ImageStats, SymbolsTable


def run(
    syms: str,
    img_dirs: List[str],
    tr_txt_table: str,
    va_txt_table: str,
    common: CommonArgs = CommonArgs(),
    train: TrainArgs = TrainArgs(),
    optimizer: OptimizerArgs = OptimizerArgs(),
    scheduler: SchedulerArgs = SchedulerArgs(),
    data: DataArgs = DataArgs(),
    trainer: TrainerArgs = TrainerArgs(),
    num_workers: Optional[int] = None,
):
    pl.seed_everything(common.seed)

    loader = ModelLoader(
        common.train_path, filename=common.model_filename, device="cpu"
    )
    # maybe load a checkpoint
    checkpoint = None
    if train.resume:
        checkpoint = loader.prepare_checkpoint(
            common.checkpoint, common.experiment_dirpath, common.monitor
        )
        trainer.max_epochs = torch.load(checkpoint)["epoch"] + train.resume
        log.info(f'Using checkpoint "{checkpoint}"')
        log.info(f"Max epochs set to {trainer.max_epochs}")

    # load the non-pytorch_lightning model
    model = loader.load()
    assert (
        model is not None
    ), "Could not find the model. Have you run pylaia-htr-create-model?"

    # prepare the symbols
    syms = SymbolsTable(syms)
    for d in train.delimiters:
        assert d in syms, f'The delimiter "{d}" is not available in the symbols file'

    # prepare the engine
    engine_module = HTREngineModule(
        model,
        [syms[d] for d in train.delimiters],
        optimizer=optimizer,
        scheduler=scheduler,
        batch_input_fn=Compose([ItemFeeder("img"), ImageFeeder()]),
        batch_target_fn=ItemFeeder("txt"),
        batch_id_fn=ItemFeeder("id"),  # Used to print image ids on exception
    )

    # prepare the data
    im_stats = ImageStats(
        stage="fit",
        tr_txt_table=tr_txt_table,
        va_txt_table=va_txt_table,
        img_dirs=img_dirs,
    )
    data_module = DataModule(
        syms=syms,
        img_dirs=img_dirs,
        tr_txt_table=tr_txt_table,
        va_txt_table=va_txt_table,
        batch_size=data.batch_size,
        min_valid_size=model.get_min_valid_image_size(im_stats.max_width)
        if im_stats.is_fixed_height
        else None,
        color_mode=data.color_mode,
        #shuffle_tr=not bool(trainer.limit_train_batches),
        shuffle_tr=True if trainer.limit_train_batches==1 else False,
        augment_tr=train.augment_training,
        stage="fit",
        num_workers=num_workers,
    )

    # prepare the training callbacks
    # TODO: save on lowest_va_wer and every k epochs https://github.com/PyTorchLightning/pytorch-lightning/issues/2908
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=common.experiment_dirpath,
        filename="{epoch}-lowest_" + common.monitor,
        monitor=common.monitor,
        verbose=True,
        save_top_k=train.checkpoint_k,
        mode="min",
        save_last=True,
    )
    checkpoint_callback.CHECKPOINT_NAME_LAST = "{epoch}-last"
    early_stopping_callback = pl.callbacks.EarlyStopping(
        monitor=common.monitor,
        patience=train.early_stopping_patience,
        verbose=True,
        mode="min",
        strict=False,  # training_step may return None
    )
    callbacks = [
        ProgressBar(refresh_rate=trainer.progress_bar_refresh_rate),
        checkpoint_callback,
        early_stopping_callback,
        checkpoint_callback,
    ]
    if train.gpu_stats:
        callbacks.append(ProgressBarGPUStats())
    if scheduler.active:
        callbacks.append(LearningRate(logging_interval="epoch"))

    # prepare the trainer
    trainer = pl.Trainer(
        default_root_dir=common.train_path,
        resume_from_checkpoint=checkpoint,
        callbacks=callbacks,
        logger=EpochCSVLogger(common.experiment_dirpath),
        checkpoint_callback=True,
        **vars(trainer),
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


def get_args(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    parser = jsonargparse.ArgumentParser(parse_as_dict=True)
    parser.add_argument(
        "--config", action=jsonargparse.ActionConfigFile, help="Configuration file"
    )
    parser.add_argument(
        "syms",
        type=str,
        help=(
            "Mapping from strings to integers. "
            "The CTC symbol must be mapped to integer 0"
        ),
    )
    parser.add_argument(
        "img_dirs",
        type=List[str],
        default=[],
        help="Directories containing segmented line images",
    )
    parser.add_argument(
        "tr_txt_table",
        type=str,
        help="Character transcription of each training image",
    )
    parser.add_argument(
        "va_txt_table",
        type=str,
        help="Character transcription of each validation image",
    )
    parser.add_class_arguments(CommonArgs, "common")
    parser.add_class_arguments(DataArgs, "data")
    parser.add_class_arguments(TrainArgs, "train")
    parser.add_function_arguments(log.config, "logging")
    parser.add_class_arguments(OptimizerArgs, "optimizer")
    parser.add_class_arguments(SchedulerArgs, "scheduler")
    parser.add_class_arguments(TrainerArgs, "trainer")

    args = parser.parse_args(argv, with_meta=False)

    args["common"] = CommonArgs(**args["common"])
    args["train"] = TrainArgs(**args["train"])
    args["data"] = DataArgs(**args["data"])
    args["optimizer"] = OptimizerArgs(**args["optimizer"])
    args["scheduler"] = SchedulerArgs(**args["scheduler"])
    args["trainer"] = TrainerArgs(**args["trainer"])

    return args


def main():
    args = get_args()
    args = common_main(args)
    run(**args)


if __name__ == "__main__":
    main()
