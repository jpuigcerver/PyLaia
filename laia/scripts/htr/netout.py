#!/usr/bin/env python3
from os.path import join
from typing import Any, Dict, List, Optional

import jsonargparse
import pytorch_lightning as pl

import laia.common.logging as log
from laia import get_installed_versions
from laia.callbacks import Netout, ProgressBar
from laia.common.arguments import CommonArgs, DataArgs, NetoutArgs, TrainerArgs
from laia.common.loader import ModelLoader
from laia.engine import Compose, DataModule, EvaluatorModule, ImageFeeder, ItemFeeder
from laia.utils.kaldi import ArchiveLatticeWriter, ArchiveMatrixWriter


def run(
    img_list: str,
    img_dirs: Optional[List[str]] = None,
    common: CommonArgs = CommonArgs(),
    data: DataArgs = DataArgs(),
    netout: NetoutArgs = NetoutArgs(),
    trainer: TrainerArgs = TrainerArgs(),
):
    loader = ModelLoader(
        common.train_path, filename=common.model_filename, device="cpu"
    )
    checkpoint = loader.prepare_checkpoint(
        common.checkpoint, common.experiment_dirpath, common.monitor
    )
    model = loader.load_by(checkpoint)
    assert (
        model is not None
    ), "Could not find the model. Have you run pylaia-htr-create-model?"

    # prepare the evaluator
    evaluator_module = EvaluatorModule(
        model,
        batch_input_fn=Compose([ItemFeeder("img"), ImageFeeder()]),
        batch_id_fn=ItemFeeder("id"),
    )

    # prepare the data
    data_module = DataModule(
        img_dirs=img_dirs,
        te_img_list=img_list,
        batch_size=data.batch_size,
        color_mode=data.color_mode,
        stage="test",
    )

    # prepare the kaldi writers
    writers = []
    if netout.matrix is not None:
        writers.append(
            ArchiveMatrixWriter(join(common.experiment_dirpath, netout.matrix))
        )
    if netout.lattice is not None:
        writers.append(
            ArchiveLatticeWriter(
                join(common.experiment_dirpath, netout.lattice),
                digits=netout.digits,
                negate=True,
            )
        )
    assert (
        writers
    ), "You did not specify any output file! Use the matrix/lattice arguments"

    # prepare the testing callbacks
    callbacks = [
        Netout(writers, output_transform=netout.output_transform),
        ProgressBar(refresh_rate=trainer.progress_bar_refresh_rate),
    ]

    # prepare the trainer
    trainer = pl.Trainer(
        default_root_dir=common.train_path,
        callbacks=callbacks,
        logger=False,
        **vars(trainer),
    )

    # run netout!
    trainer.test(evaluator_module, datamodule=data_module, verbose=False)


def get_args(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    parser = jsonargparse.ArgumentParser(parse_as_dict=True)
    parser.add_argument(
        "--config", action=jsonargparse.ActionConfigFile, help="Configuration file"
    )
    parser.add_argument(
        "img_list",
        type=str,
        help=(
            "File containing the images to decode. Each image is expected to be in one "
            'line. Lines starting with "#" will be ignored. Lines can be filepaths '
            '(e.g. "/tmp/img.jpg") or filenames of images present in --img_dirs (e.g. '
            "img.jpg). The filename extension is optional and case insensitive"
        ),
    )
    parser.add_argument(
        "--img_dirs",
        type=Optional[List[str]],
        default=None,
        help=(
            "Directories containing word images. "
            "Optional if `img_list` contains filepaths"
        ),
    )
    parser.add_class_arguments(CommonArgs, "common")
    parser.add_class_arguments(DataArgs, "data")
    parser.add_function_arguments(log.config, "logging")
    parser.add_class_arguments(NetoutArgs, "netout")
    parser.add_class_arguments(TrainerArgs, "trainer")

    args = parser.parse_args(argv, with_meta=False)

    args["common"] = CommonArgs(**args["common"])
    args["data"] = DataArgs(**args["data"])
    args["netout"] = NetoutArgs(**args["netout"])
    args["trainer"] = TrainerArgs(**args["trainer"])

    return args


def main():
    args = get_args()
    del args["config"]
    # configure logging
    logging = args.pop("logging")
    if logging["filepath"] is not None:
        logging["filepath"] = join(
            args["common"].experiment_dirpath, logging["filepath"]
        )
    log.config(**logging)
    log.info(f"Arguments: {args}")
    log.info(f"Installed: {get_installed_versions()}")
    run(**args)


if __name__ == "__main__":
    main()
