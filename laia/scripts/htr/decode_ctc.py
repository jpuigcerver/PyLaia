#!/usr/bin/env python3
import argparse
from os.path import join

import pytorch_lightning as pl

import laia.common.logging as log
from laia import get_installed_versions
from laia.callbacks import Decode, ProgressBar, Segmentation
from laia.common.arguments import LaiaParser, add_lightning_args, group_to_namespace
from laia.common.loader import ModelLoader
from laia.engine import Compose, DataModule, EvaluatorModule, ImageFeeder, ItemFeeder
from laia.utils import SymbolsTable


def run(args: argparse.Namespace):
    log.info(f"Installed: {get_installed_versions()}")

    loader = ModelLoader(args.train_path, filename=args.model_filename, device="cpu")
    checkpoint = loader.prepare_checkpoint(
        args.checkpoint, join(args.train_path, args.experiment_dirname), args.monitor
    )
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

    # prepare the symbols
    syms = SymbolsTable(args.syms)

    # prepare the data
    data_module = DataModule(
        img_dirs=args.img_dirs,
        color_mode=args.color_mode,
        batch_size=args.batch_size,
        te_img_list=args.img_list,
        syms=syms,
        stage="test",
    )

    # prepare the testing callbacks
    callbacks = [
        ProgressBar(refresh_rate=args.lightning.progress_bar_refresh_rate),
        Segmentation(
            syms,
            segmentation=args.print_segmentation,
            input_space=args.input_space,
            separator=args.separator,
            include_img_ids=args.include_img_ids,
        )
        if bool(args.print_segmentation)
        else Decode(
            syms=syms,
            use_symbols=args.use_symbols,
            input_space=args.input_space,
            output_space=args.output_space,
            convert_spaces=args.convert_spaces,
            join_str=args.join_str,
            separator=args.separator,
            include_img_ids=args.include_img_ids,
        ),
    ]

    # prepare the trainer
    trainer = pl.Trainer(
        default_root_dir=args.train_path,
        callbacks=callbacks,
        **vars(args.lightning),
    )

    # decode!
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
        "syms",
        type=argparse.FileType("r"),
        help="Symbols table mapping from strings to integers",
    ).add_argument(
        "img_list",
        type=argparse.FileType("r"),
        help="File containing images to decode. Doesn't require the extension",
    ).add_argument(
        "img_dirs",
        type=str,
        nargs="*",
        help=(
            "Directory containing word images. "
            "Optional if img_list contains whole paths"
        ),
    ).add_argument(
        "--include_img_ids",
        type=pl.utilities.parsing.str_to_bool,
        nargs="?",
        const=True,
        default=True,
        help="Include the associated image ids in the output",
    ).add_argument(
        "--separator",
        type=str,
        default=" ",
        help="Use this string as the separator between the ids and the output",
    ).add_argument(
        "--join_str", type=str, default=None, help="Join the output using this"
    ).add_argument(
        "--use_symbols",
        type=pl.utilities.parsing.str_to_bool,
        nargs="?",
        const=True,
        default=True,
        help="Print the output with symbols instead of numbers",
    ).add_argument(
        "--convert_spaces",
        type=pl.utilities.parsing.str_to_bool,
        nargs="?",
        const=True,
        default=False,
        help="Whether or not to convert spaces",
    ).add_argument(
        "--input_space",
        type=str,
        default="<space>",
        help="Input space symbol to replace",
    ).add_argument(
        "--output_space", type=str, default=" ", help="Output space symbol"
    ).add_argument(
        "--print_segmentation",
        type=str,
        default=None,
        choices=["char", "word"],
        help="Print output with the corresponding segmentation",
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
