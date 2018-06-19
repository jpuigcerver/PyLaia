#!/usr/bin/env python
import os
from argparse import FileType

import torch

import laia.utils
from laia.engine.engine import EPOCH_START, EPOCH_END
from laia.engine.phoc_engine_wrapper import PHOCEngineWrapper
from laia.hooks import Hook, HookCollection, action, Action
from laia.hooks.conditions import GEqThan, Highest, MultipleOf, Any
from laia.losses.dortmund_bce_loss import DortmundBCELoss
from laia.models.kws.dortmund_phocnet import DortmundPHOCNet
from laia.plugins import CheckpointSaver, ModelCheckpointSaver, RollingSaver
from laia.plugins.arguments import add_argument, add_defaults, args
from laia.plugins.arguments_types import str2bool
from laia.utils.dortmund_image_to_tensor import DortmundImageToTensor

logger = laia.logging.get_logger("laia.egs.washington.train_phoc")
laia.logging.get_logger("laia.hooks.conditions.multiple_of").setLevel(
    laia.logging.WARNING
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
        "--phoc_levels",
        type=int,
        default=[1, 2, 3, 4, 5],
        nargs="+",
        help="PHOC levels used to encode the transcript",
    )
    add_argument(
        "--tpp_levels",
        type=int,
        default=[1, 2, 3, 4, 5],
        nargs="*",
        help="Temporal Pyramid Pooling levels",
    )
    add_argument(
        "--spp_levels",
        type=int,
        default=None,
        nargs="*",
        help="Spatial Pyramid Pooling levels",
    )
    add_argument(
        "--exclude_words_ap",
        type=FileType("r"),
        help="List of words to exclude in the Average Precision " "computation",
    )
    add_argument(
        "--use_adam_optim",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="If true, use Adam optimizer instead of SGD",
    )
    add_argument("--use_new_phoc", action="store_true")
    add_argument("syms", help="Symbols table mapping from strings to integers")
    add_argument("tr_img_dir", help="Directory containing word images")
    add_argument("tr_txt_table", help="Character transcriptions of each training image")
    add_argument(
        "va_txt_table", help="Character transcriptions of each validation image"
    )
    args = args()

    laia.random.manual_seed(args.seed)

    syms = laia.utils.SymbolsTable(args.syms)

    phoc_size = sum(args.phoc_levels) * len(syms)
    model = DortmundPHOCNet(
        phoc_size=phoc_size, tpp_levels=args.tpp_levels, spp_levels=args.spp_levels
    )
    if args.load_checkpoint:
        model_ckpt = torch.load(args.load_checkpoint)
        model.load_state_dict(model_ckpt)
    model = model.cuda(args.gpu - 1) if args.gpu > 0 else model.cpu()
    logger.info("PHOC embedding size = {}", phoc_size)
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

    # If --use_distortions is given, apply the same affine distortions used by
    # Dortmund University.
    if args.use_distortions:
        tr_img_transform = DortmundImageToTensor()
    else:
        tr_img_transform = laia.utils.ImageToTensor()

    # Training data
    tr_ds = laia.data.TextImageFromTextTableDataset(
        args.tr_txt_table, args.tr_img_dir, img_transform=tr_img_transform
    )
    if args.train_samples_per_epoch is None:
        tr_ds_loader = torch.utils.data.DataLoader(
            tr_ds,
            batch_size=1,
            num_workers=8,
            shuffle=True,
            collate_fn=laia.data.PaddingCollater(
                {"img": [1, None, None]}, sort_key=lambda x: -x["img"].size(2)
            ),
        )
    else:
        tr_ds_loader = torch.utils.data.DataLoader(
            tr_ds,
            batch_size=1,
            num_workers=8,
            sampler=laia.data.FixedSizeSampler(tr_ds, args.train_samples_per_epoch),
            collate_fn=laia.data.PaddingCollater(
                {"img": [1, None, None]}, sort_key=lambda x: -x["img"].size(2)
            ),
        )

    # Validation data
    va_ds = laia.data.TextImageFromTextTableDataset(
        args.va_txt_table, args.tr_img_dir, img_transform=laia.utils.ImageToTensor()
    )
    if args.valid_samples_per_epoch is None:
        va_ds_loader = torch.utils.data.DataLoader(
            va_ds,
            batch_size=1,
            num_workers=8,
            collate_fn=laia.data.PaddingCollater(
                {"img": [1, None, None]}, sort_key=lambda x: -x["img"].size(2)
            ),
        )
    else:
        va_ds_loader = torch.utils.data.DataLoader(
            va_ds,
            batch_size=1,
            num_workers=8,
            sampler=laia.data.FixedSizeSampler(va_ds, args.valid_samples_per_epoch),
            collate_fn=laia.data.PaddingCollater(
                {"img": [1, None, None]}, sort_key=lambda x: -x["img"].size(2)
            ),
        )

    trainer = laia.engine.Trainer(
        model=model,
        criterion=DortmundBCELoss(),
        optimizer=optimizer,
        data_loader=tr_ds_loader,
        progress_bar="Train" if args.show_progress_bar else False,
    )
    trainer.iterations_per_update = args.iterations_per_update

    evaluator = laia.engine.Evaluator(
        model=model,
        data_loader=va_ds_loader,
        progress_bar="Valid" if args.show_progress_bar else False,
    )

    if args.exclude_words_ap:
        exclude_words_ap = set([x.strip() for x in args.exclude_words_ap])
    else:
        exclude_words_ap = None

    engine_wrapper = PHOCEngineWrapper(
        symbols_table=syms,
        phoc_levels=args.phoc_levels,
        train_engine=trainer,
        valid_engine=evaluator,
        gpu=args.gpu,
        exclude_labels=exclude_words_ap,
        use_new_phoc=args.use_new_phoc,
    )

    highest_gap_saver = RollingSaver(
        ModelCheckpointSaver(
            CheckpointSaver(
                os.path.join(args.train_path, "model.ckpt-highest-valid-gap")
            ),
            model,
        ),
        keep=3,
    )

    highest_map_saver = RollingSaver(
        ModelCheckpointSaver(
            CheckpointSaver(
                os.path.join(args.train_path, "model.ckpt-highest-valid-map")
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
                Highest(engine_wrapper.valid_ap(), key=0, name="Highest gAP"),
                Action(save_ckpt, saver=highest_gap_saver),
            ),
            Hook(
                Highest(engine_wrapper.valid_ap(), key=1, name="Highest mAP"),
                Action(save_ckpt, saver=highest_map_saver),
            ),
            Hook(model_saver_when, Action(save_ckpt, saver=model_saver)),
        ),
    )

    if args.continue_epoch:
        trainer._epochs = args.continue_epoch

    # Launch training
    engine_wrapper.run()
