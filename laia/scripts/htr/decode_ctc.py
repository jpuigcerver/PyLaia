#!/usr/bin/env python3
from typing import Any, Dict, List, Optional

import jsonargparse
import pytorch_lightning as pl

import laia.common.logging as log
from laia.callbacks import Decode, ProgressBar, Segmentation
from laia.common.arguments import CommonArgs, DataArgs, DecodeArgs, TrainerArgs
from laia.common.loader import ModelLoader
from laia.engine import Compose, DataModule, EvaluatorModule, ImageFeeder, ItemFeeder
from laia.scripts.htr import common_main
from laia.utils import SymbolsTable


def run(
    syms: str,
    img_list: str,
    img_dirs: Optional[List[str]] = None,
    common: CommonArgs = CommonArgs(),
    data: DataArgs = DataArgs(),
    decode: DecodeArgs = DecodeArgs(),
    trainer: TrainerArgs = TrainerArgs(),
):
    loader = ModelLoader(
        common.train_path, filename=common.model_filename, device="cpu"
    )
    checkpoint = loader.prepare_checkpoint(
        common.checkpoint,
        common.experiment_dirpath,
        common.monitor,
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

    # prepare the symbols
    syms = SymbolsTable(syms)

    # prepare the data
    data_module = DataModule(
        syms=syms,
        img_dirs=img_dirs,
        te_img_list=img_list,
        batch_size=data.batch_size,
        color_mode=data.color_mode,
        stage="test",
    )

    # prepare the testing callbacks
    callbacks = [
        ProgressBar(refresh_rate=trainer.progress_bar_refresh_rate),
        Segmentation(
            syms,
            segmentation=decode.segmentation,
            input_space=decode.input_space,
            separator=decode.separator,
            include_img_ids=decode.include_img_ids,
        )
        if bool(decode.segmentation)
        else Decode(
            syms=syms,
            use_symbols=decode.use_symbols,
            input_space=decode.input_space,
            output_space=decode.output_space,
            convert_spaces=decode.convert_spaces,
            join_string=decode.join_string,
            separator=decode.separator,
            include_img_ids=decode.include_img_ids,
            print_line_confidence_scores=decode.print_line_confidence_scores,
            print_word_confidence_scores=decode.print_word_confidence_scores,
        ),
    ]

    # prepare the trainer
    trainer = pl.Trainer(
        default_root_dir=common.train_path,
        callbacks=callbacks,
        logger=False,
        **vars(trainer),
    )

    # decode!
    trainer.test(evaluator_module, datamodule=data_module, verbose=False)


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
    parser.add_class_arguments(DecodeArgs, "decode")
    parser.add_class_arguments(TrainerArgs, "trainer")

    args = parser.parse_args(argv, with_meta=False)

    args["common"] = CommonArgs(**args["common"])
    args["data"] = DataArgs(**args["data"])
    args["decode"] = DecodeArgs(**args["decode"])
    args["trainer"] = TrainerArgs(**args["trainer"])

    return args


def main():
    args = get_args()
    args = common_main(args)
    run(**args)


if __name__ == "__main__":
    main()
