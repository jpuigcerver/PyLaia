#!/usr/bin/env python3
import argparse
from os.path import join

import pytorch_lightning as pl

import laia.common.logging as log
from laia import get_installed_versions
from laia.callbacks import Netout, ProgressBar
from laia.common.arguments import LaiaParser, add_lightning_args, group_to_namespace
from laia.common.loader import ModelLoader
from laia.engine import Compose, DataModule, EvaluatorModule, ImageFeeder, ItemFeeder
from laia.utils.kaldi import ArchiveLatticeWriter, ArchiveMatrixWriter


def run(args: argparse.Namespace):
    log.info(f"Installed: {get_installed_versions()}")

    exp_dirpath = join(args.train_path, args.experiment_dirname)
    loader = ModelLoader(args.train_path, filename=args.model_filename, device="cpu")
    checkpoint = loader.prepare_checkpoint(args.checkpoint, exp_dirpath, args.monitor)
    model = loader.load_by(checkpoint)
    if model is None:
        log.error('Could not find the model. Have you run "pylaia-htr-create-model"?')
        exit(1)

    # prepare the evaluator
    evaluator_module = EvaluatorModule(
        model,
        batch_input_fn=Compose([ItemFeeder("img"), ImageFeeder()]),
        batch_id_fn=ItemFeeder("id"),
    )

    # prepare the data
    data_module = DataModule(
        img_dirs=args.img_dirs,
        color_mode=args.color_mode,
        batch_size=args.batch_size,
        te_img_list=args.img_list,
        stage="test",
    )

    # prepare the kaldi writers
    writers = []
    if args.matrix is not None:
        writers.append(ArchiveMatrixWriter(join(exp_dirpath, args.matrix)))
    if args.lattice is not None:
        writers.append(
            ArchiveLatticeWriter(
                join(exp_dirpath, args.lattice), digits=args.digits, negate=True
            )
        )
    if not writers:
        log.error("You did not specify any output file! Use --matrix and/or --lattice")
        exit(1)

    # prepare the testing callbacks
    callbacks = [
        Netout(writers, output_transform=args.output_transform),
        ProgressBar(refresh_rate=args.lightning.progress_bar_refresh_rate),
    ]

    # prepare the trainer
    trainer = pl.Trainer(
        default_root_dir=args.train_path,
        callbacks=callbacks,
        **vars(args.lightning),
    )

    # run netout!
    trainer.test(evaluator_module, datamodule=data_module, verbose=False)


def get_args() -> argparse.Namespace:
    parser = LaiaParser().add_defaults(
        "batch_size",
        "train_path",
        "monitor",
        "checkpoint",
        "model_filename",
        "experiment_dirname",
        "color_mode",
    )
    parser.add_argument(
        "img_list",
        type=argparse.FileType("r"),
        help=(
            "File containing the images to decode. Each image is expected to be in one "
            'line. Lines starting with "#" will be ignored. Lines can be filepaths '
            '(e.g. "/tmp/img.jpg") or filenames of images present in --img_dirs (e.g. '
            "img.jpg). The filename extension is optional and case insensitive"
        ),
    ).add_argument(
        "img_dirs",
        type=str,
        nargs="*",
        help=(
            "Directory containing word images. "
            "Optional if img_list contains filepaths"
        ),
    ).add_argument(
        "--output_transform",
        type=str,
        default=None,
        choices=["softmax", "log_softmax"],
        help=(
            "Apply this transformation at the end of the model. "
            'For instance, use "softmax" to get posterior probabilities '
            "as the output of the model"
        ),
    ).add_argument(
        "--matrix",
        type=str,
        default=None,
        help=(
            "Path of the Kaldi's archive containing the output matrices "
            "(one for each sample), where each row represents a timestep and "
            "each column represents a CTC label"
        ),
    ).add_argument(
        "--lattice",
        type=str,
        default=None,
        help=(
            "Path of the Kaldi's archive containing the output lattices"
            "(one for each sample), representing the CTC output"
        ),
    ).add_argument(
        "--digits",
        type=int,
        default=10,
        help="Number of digits to be used for formatting",
    )

    # Add lightning default arguments to a group
    pl_group = parser.parser.add_argument_group(title="pytorch-lightning arguments")
    pl_group = add_lightning_args(
        pl_group,
        blocklist=[
            "default_root_dir",
            "auto_lr_find",
            # TODO: support this
            "auto_scale_batch_size",
        ],
    )

    args = parser.parse_args()

    # Move lightning default arguments to their own namespace
    args = group_to_namespace(args, pl_group, "lightning")

    return args


def main():
    run(get_args())


if __name__ == "__main__":
    main()
