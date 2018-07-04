#!/usr/bin/env python
import os

import torch

import laia.utils
from laia.engine.engine import EPOCH_START, EPOCH_END
from laia.engine.feeders import ImageFeeder, ItemFeeder
from laia.experiments.htr_experiment import HTRExperiment
from laia.engine.trainer import Trainer
from laia.hooks import Hook, HookCollection, action, Action
from laia.hooks.conditions import Any, GEqThan, Lowest, MultipleOf
from laia.models.htr.dortmund_crnn import DortmundCRNN
from laia.plugins.arguments import add_argument, add_defaults, args
from laia.plugins.arguments_types import str2bool
from laia.plugins.saver import CheckpointSaver, ModelCheckpointSaver, RollingSaver
from laia.utils.dortmund_image_to_tensor import DortmundImageToTensor

logger = laia.common.logging.get_logger("laia.egs.washington.train_ctc")
laia.common.logging.get_logger("laia.hooks.conditions.multiple_of").setLevel(
    laia.common.logging.WARNING
)

if __name__ == "__main__":
    add_defaults(
        "gpu",
        "max_epochs",
        "max_updates",
        "train_samples_per_epoch",
        "valid_samples_per_epoch",
        "seed",
        "train_path",
        # Override default values for these arguments, but use the
        # same help/checks:
        batch_size=1,
        learning_rate=0.0001,
        momentum=0.9,
        num_rolling_checkpoints=5,
        iterations_per_update=10,
        save_checkpoint_interval=5,
        show_progress_bar=True,
        use_distortions=True,
        weight_l2_penalty=0.00005,
    )
    add_argument("--load_checkpoint", type=str, help="Path to the checkpoint to load.")
    add_argument("--continue_epoch", type=int)
    add_argument(
        "--image_sequencer",
        type=str,
        default="avgpool-16",
        help="Average adaptive pooling of the images before the LSTM layers",
    )
    add_argument(
        "--use_adam_optim",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="If true, use Adam optimizer instead of SGD",
    )
    add_argument(
        "--keep_padded_tensors",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="If true, use Adam optimizer instead of SGD",
    )
    add_argument("--lstm_hidden_size", type=int, default=128)
    add_argument("--lstm_num_layers", type=int, default=1)
    add_argument("--min_size", type=int, default=None)
    add_argument("--dropout", type=float, default=0.5)
    add_argument("syms", help="Symbols table mapping from strings to integers")
    add_argument("tr_img_dir", help="Directory containing word images")
    add_argument("tr_txt_table", help="Character transcriptions of each training image")
    add_argument(
        "va_txt_table", help="Character transcriptions of each validation image"
    )

    args = args()
    laia.common.random.manual_seed(args.seed)

    syms = laia.utils.SymbolsTable(args.syms)

    # If we use --keep_padded_tensors=false, we need to update the
    # iteration_per_update, since the maximum batch_size that we can use is 1
    if not args.keep_padded_tensors:
        args.iterations_per_update *= args.batch_size
        args.batch_size = 1

    # If --use_distortions is given, apply the same affine distortions used by
    # Dortmund University.
    if args.use_distortions:
        tr_img_transform = DortmundImageToTensor(
            min_width=args.min_size, min_height=args.min_size
        )
    else:
        tr_img_transform = laia.utils.ImageToTensor(
            min_width=args.min_size, min_height=args.min_size
        )

    # Training data
    tr_ds = laia.data.TextImageFromTextTableDataset(
        args.tr_txt_table,
        args.tr_img_dir,
        img_transform=tr_img_transform,
        txt_transform=laia.utils.TextToTensor(syms),
    )
    if args.train_samples_per_epoch is None:
        tr_ds_loader = laia.data.ImageDataLoader(
            tr_ds,
            image_channels=1,
            batch_size=args.batch_size,
            num_workers=8,
            shuffle=True,
        )
    else:
        tr_ds_loader = laia.data.ImageDataLoader(
            tr_ds,
            image_channels=1,
            batch_size=args.batch_size,
            num_workers=8,
            sampler=laia.data.FixedSizeSampler(tr_ds, args.train_samples_per_epoch),
        )

    # Validation data
    va_ds = laia.data.TextImageFromTextTableDataset(
        args.va_txt_table,
        args.tr_img_dir,
        img_transform=laia.utils.ImageToTensor(
            min_width=args.min_size, min_height=args.min_size
        ),
        txt_transform=laia.utils.TextToTensor(syms),
    )
    if args.valid_samples_per_epoch is None:
        va_ds_loader = laia.data.ImageDataLoader(
            va_ds,
            image_channels=1,
            batch_size=args.batch_size,
            num_workers=8,
            shuffle=True,
        )
    else:
        va_ds_loader = laia.data.ImageDataLoader(
            va_ds,
            image_channels=1,
            batch_size=args.batch_size,
            num_workers=8,
            sampler=laia.data.FixedSizeSampler(va_ds, args.valid_samples_per_epoch),
        )

    model = DortmundCRNN(
        num_outputs=len(syms),
        lstm_hidden_size=args.lstm_hidden_size,
        lstm_num_layers=args.lstm_num_layers,
        sequencer=args.image_sequencer,
        dropout=args.dropout,
    )

    if args.load_checkpoint:
        model_ckpt = torch.load(args.load_checkpoint)
        model.load_state_dict(model_ckpt)

    model = model.cuda(args.gpu - 1) if args.gpu > 0 else model.cpu()
    logger.info(
        "Model has {} parameters",
        sum(param.data.numel() for param in model.parameters()),
    )

    if args.use_adam_optim:
        logger.info(
            "Using ADAM optimizer with learning rate = {:g} and weight decay = {:g}",
            args.learning_rate,
            args.weight_l2_penalty,
        )
        optimizer = torch.optim.Adam(
            params=model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_l2_penalty,
        )
    else:
        logger.info(
            "Using SGD optimizer with learning rate = {:g}, momentum = {:g} and "
            "weight decay = {:g}",
            args.learning_rate,
            args.momentum,
            args.weight_l2_penalty,
        )
        optimizer = torch.optim.SGD(
            params=model.parameters(),
            lr=args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_l2_penalty,
        )
    parameters = {
        "model": model,
        "criterion": None,  # Set automatically by HtrEngineWrapper
        "optimizer": optimizer,
        "data_loader": tr_ds_loader,
        "batch_input_fn": ImageFeeder(
            device=args.gpu,
            keep_padded_tensors=args.keep_padded_tensors,
            parent_feeder=ItemFeeder("img")
        ),
        "batch_target_fn": ItemFeeder("txt"),
        "batch_id_fn": ItemFeeder("id"),  # Print image ids on exception
        "progress_bar": "Train" if args.show_progress_bar else False,
    }
    trainer = Trainer(**parameters)
    trainer.iterations_per_update = args.iterations_per_update

    evaluator = laia.engine.Evaluator(
        model=model,
        data_loader=va_ds_loader,
        batch_input_fn=ImageFeeder(
            device=args.gpu,
            keep_padded_tensors=args.keep_padded_tensors,
            parent_feeder=ItemFeeder("img")
        ),
        batch_target_fn=ItemFeeder("txt"),
        batch_id_fn=ItemFeeder("id"),  # Print image ids on exception
        progress_bar="Valid" if args.show_progress_bar else False,
    )

    engine_wrapper = HTRExperiment(trainer, evaluator)
    engine_wrapper.set_word_delimiters([])

    lowest_cer_saver = RollingSaver(
        ModelCheckpointSaver(
            CheckpointSaver(
                os.path.join(args.train_path, "model.ckpt-lowest-valid-cer")
            ),
            model,
        ),
        keep=3,
    )

    lowest_wer_saver = RollingSaver(
        ModelCheckpointSaver(
            CheckpointSaver(
                os.path.join(args.train_path, "model.ckpt-lowest-valid-wer")
            ),
            model,
        ),
        keep=3,
    )

    model_saver = RollingSaver(
        ModelCheckpointSaver(
            CheckpointSaver(os.path.join(args.train_path, "model.ckpt")), model
        ),
        keep=args.num_rolling_checkpoints,
    )

    @action
    def save_ckpt(saver, epoch):
        saver.save(suffix=epoch)

    # Set hooks
    if args.max_epochs and args.max_epochs > 0:
        trainer.add_hook(
            EPOCH_START, Hook(GEqThan(trainer.epochs, args.max_epochs), trainer.stop)
        )
        model_saver_when = Any(
            GEqThan(trainer.epochs, args.max_epochs - args.num_rolling_checkpoints),
            MultipleOf(trainer.epochs, args.save_checkpoint_interval),
        )
    else:
        model_saver_when = MultipleOf(trainer.epochs, args.save_checkpoint_interval)

    trainer.add_hook(
        EPOCH_END,
        HookCollection(
            Hook(
                Lowest(engine_wrapper.valid_cer(), name="Lowest CER"),
                Action(save_ckpt, saver=lowest_cer_saver),
            ),
            Hook(
                Lowest(engine_wrapper.valid_wer(), name="Lowest WER"),
                Action(save_ckpt, lowest_wer_saver),
            ),
            Hook(model_saver_when, Action(save_ckpt, saver=model_saver)),
        ),
    )

    if args.continue_epoch:
        trainer._epochs = args.continue_epoch

    # Launch training
    engine_wrapper.run()
