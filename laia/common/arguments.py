import inspect
from dataclasses import dataclass, field, make_dataclass
from distutils.version import LooseVersion
from enum import Enum
from os.path import join
from typing import Any, List, Optional, Tuple, Type, Union

import pytorch_lightning as pl
import torch
from jsonargparse.typing import (
    ClosedUnitInterval,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    restricted_number_type,
)

GeNeg1Int = restricted_number_type(None, int, (">=", -1))


class Monitor(str, Enum):
    va_loss = "va_loss"
    va_cer = "va_cer"
    va_wer = "va_wer"


@dataclass
class CommonArgs:
    """Common arguments

    Args:
        seed: Seed for random number generators
        train_path: Save any files in this location
        model_filename: Filename of the model
        experiment_dirname: Directory name of the experiment.
            It will be created inside the `train_path`
        checkpoint: Any of:
            1 - None: load the best checkpoint with respect to the `monitor`,
                checkpoints will be searched in the `experiment_dirname` directory
            2 - A filepath: the filepath to the checkpoint (e.g. "/tmp/model.ckpt")
            3 - A filename: the filename of the checkpoint to load inside the
                `experiment_dirname` directory (e.g. "model.ckpt")
            4 - A glob pattern: e.g. "epoch=*.ckpt" will load the checkpoint with
                the highest epoch number, globbing will be done inside the
                `experiment_dirname` directory
        monitor: Metric to monitor for early stopping and checkpointing.
    """

    seed: int = 74565  # 0x12345
    train_path: str = ""
    model_filename: str = "model"
    experiment_dirname: str = "experiment"
    monitor: Monitor = Monitor.va_cer
    checkpoint: Optional[str] = None

    def __post_init__(self):
        self.experiment_dirpath = join(self.train_path, self.experiment_dirname)


@dataclass
class CreateCRNNArgs:
    """Create LaiaCRNN arguments

    Args:
        num_input_channels: Number of channels of the input images
        vertical_text: Whether the text is written vertically
        cnn_num_features: Number of features in each convolutional layer
        cnn_kernel_size: Kernel size of each convolutional layer.
             (e.g. [n,n,...] or [[h1,w1],[h2,w2],...])
        cnn_stride: Stride of each convolutional layer.
             (e.g. [n,n,...] or [[h1,w1],[h2,w2],...])
        cnn_dilation: Spacing between each convolutional layer kernel elements.
             (e.g. [n,n,...] or [[h1,w1],[h2,w2],...])
        cnn_activation: Type of activation function in each convolutional layer.
            From `torch.nn`
        cnn_poolsize: MaxPooling size after each convolutional layer.
             (e.g. [n,n,...] or [[h1,w1],[h2,w2],...])
        cnn_dropout: Dropout probability at the input of each convolutional layer
        cnn_batchnorm: Whether to do batch normalization before the activation in
            each convolutional layer
        use_masks: Whether to apply a zero mask after each convolution and
            non-linear activation
        rnn_layers: Number of recurrent layers
        rnn_units: Number of units in each recurrent layer
        rnn_dropout: Dropout probability at the input of each recurrent layer
        rnn_type: Type of recurrent layer. From `torch.nn`
        lin_dropout: Dropout probability at the input of the final linear layer
    """

    num_input_channels: PositiveInt = 1
    vertical_text: bool = False
    cnn_num_features: List[PositiveInt] = field(
        default_factory=lambda: [16, 16, 32, 32]
    )
    cnn_kernel_size: List[Union[PositiveInt, List[PositiveInt]]] = field(
        default_factory=lambda: [3] * 4
    )
    cnn_stride: List[Union[PositiveInt, List[PositiveInt]]] = field(
        default_factory=lambda: [1] * 4
    )
    cnn_dilation: List[Union[PositiveInt, List[PositiveInt]]] = field(
        default_factory=lambda: [1] * 4
    )
    cnn_activation: List[str] = field(default_factory=lambda: ["LeakyReLU"] * 4)
    cnn_poolsize: List[Union[NonNegativeInt, List[NonNegativeInt]]] = field(
        default_factory=lambda: [2, 2, 2, 0]
    )
    cnn_dropout: List[ClosedUnitInterval] = field(default_factory=lambda: [0.0] * 4)
    cnn_batchnorm: List[bool] = field(default_factory=lambda: [False] * 4)
    use_masks: bool = False
    rnn_layers: PositiveInt = 3
    rnn_units: PositiveInt = 256
    rnn_dropout: ClosedUnitInterval = 0.5
    rnn_type: str = "LSTM"
    lin_dropout: ClosedUnitInterval = 0.5

    def __post_init__(self):
        dimensions = map(
            len,
            (
                self.cnn_num_features,
                self.cnn_kernel_size,
                self.cnn_stride,
                self.cnn_dilation,
                self.cnn_activation,
                self.cnn_poolsize,
                self.cnn_dropout,
                self.cnn_batchnorm,
            ),
        )
        if len(set(dimensions)) != 1:
            raise ValueError("Wrong cnn layer dimensions")
        self.cnn_kernel_size = self.parse_parameter(self.cnn_kernel_size)
        self.cnn_stride = self.parse_parameter(self.cnn_stride)
        self.cnn_dilation = self.parse_parameter(self.cnn_dilation)
        self.cnn_poolsize = self.parse_parameter(self.cnn_poolsize)
        if not all(hasattr(torch.nn, act) for act in set(self.cnn_activation)):
            raise ValueError(
                f"Could not find all cnn activations {self.cnn_activation} in `torch.nn`"
            )
        if not hasattr(torch.nn, self.rnn_type):
            raise ValueError(f"Could not find RNN type {self.rnn_type} in `torch.nn`")

    @staticmethod
    def parse_parameter(
        layers: List[Union[int, List[int]]], dim: int = 2
    ) -> List[List[int]]:
        parsed = []
        for l in layers:
            if isinstance(l, int):
                parsed.append([l] * dim)
            elif isinstance(l, (tuple, list)):
                if not all(isinstance(v, int) for v in l):
                    raise ValueError(f"An element of {layers} is not an int")
                if len(l) != dim:
                    raise ValueError(
                        f"The given input {layers} does not "
                        f"match the given dimensions {dim}"
                    )
                parsed.append(list(l))
            else:
                raise ValueError(f"{l} ({type(l)}) is neither a tuple nor an int")
        return parsed


@dataclass
class DataArgs:
    """Data arguments

    Args:
        batch_size: Batch size
        color_mode: L (grayscale): 1 channel, RGB: 3 channels, RGBA: 4 channels
    """

    class ColorMode(str, Enum):
        L = "L"
        RGB = "RGB"
        RGBA = "RGBA"

    batch_size: PositiveInt = 8
    color_mode: ColorMode = ColorMode.L


@dataclass
class TrainArgs:
    """Train arguments

    Args:
        delimiters: List of symbols representing the word delimiters.
        checkpoint_k:
            -1: all models are saved.
            0: no models are saved.
            k: the best k models will be saved
        resume: Whether to resume training with a checkpoint. See `checkpoint`.
            If a number, resumes training for this number of epochs.
        early_stopping_patience: Number of validation epochs with no improvement
            after which training will be stopped
        gpu_stats: Whether to include GPU stats in the training progress bar
        augment_training: Whether to use dynamic distortions to augment
            the training data
    """

    delimiters: Optional[List[str]] = field(default_factory=lambda: ["<space>"])
    checkpoint_k: GeNeg1Int = 3
    resume: Union[bool, NonNegativeInt] = False
    early_stopping_patience: NonNegativeInt = 20
    gpu_stats: bool = False
    augment_training: bool = False


@dataclass
class OptimizerArgs:
    """Optimizer arguments

    Args:
        name: Optimization algorithm
        learning_rate: Learning rate
        momentum: Momentum
        weight_l2_penalty: Apply this L2 weight penalty to the loss function
        nesterov: Whether to use Nesterov momentum
    """

    class Name(str, Enum):
        SGD = "SGD"
        RMSProp = "RMSProp"
        Adam = "Adam"

    name: Name = Name.RMSProp
    learning_rate: PositiveFloat = 0.0005
    momentum: NonNegativeFloat = 0.0
    weight_l2_penalty: NonNegativeFloat = 0.0
    nesterov: bool = False


@dataclass
class SchedulerArgs:
    """Scheduler arguments

    Args:
        active: Whether to use an on-plateau learning rate scheduler
        monitor: Metric for the scheduler to monitor
        patience: Number of epochs with no improvement after which
            learning rate will be reduced
        factor: Factor by which the learning rate will be reduced
    """

    active: bool = False
    monitor: Monitor = Monitor.va_loss
    patience: NonNegativeInt = 5
    factor: float = 0.1


def __get_trainer_fields() -> List[Tuple[str, Type, Any]]:
    sig = inspect.signature(pl.Trainer.__init__)
    parameters = list(sig.parameters.values())
    parameters = parameters[1:]  # exclude self
    # the following are manually set already in the scripts
    blocklist = (
        "checkpoint_callback",
        "default_root_dir",
        "resume_from_checkpoint",
        "log_gpu_memory",
        "logger",
        "callbacks",
        # TODO: support these 2
        "auto_scale_batch_size",
        "auto_lr_find",
    )
    parameters = [p for p in parameters if p.name not in blocklist]
    return [(p.name, p.annotation, p.default) for p in parameters]


@dataclass
class TrainerArgs(make_dataclass("", __get_trainer_fields())):
    __doc__ = pl.Trainer.__init__.__doc__

    def __post_init__(self):
        if (
            LooseVersion(torch.__version__) < LooseVersion("1.7.0")
            and self.precision != 32
        ):
            raise ValueError(
                "AMP requires torch>=1.7.0. Additionally, only "
                "fixed height models are currently supported"
            )


@dataclass
class DecodeArgs:
    """Decode arguments

    Args:
        include_img_ids: Include the associated image ids in the
            decoding/segmentation output
        separator: Use this as the separator between the ids and the
            decoding/segmentation output
        join_string: Join the decoding output using this string
        use_symbols: Convert the decoding output to symbols instead of numbers
        convert_spaces: Whether or not to convert spaces
        input_space: For decoding, if `convert_spaces` is set, space symbol
            to be replaced (`input_space` -> `output_space`).
            For word segmentation, symbol used to convert the sequences of
            characters into sentences of words
        output_space: Output space symbol for decoding.
            (`input_space` -> `output_space`)
        segmentation: Print this kind of segmentation instead of decoding.
    """

    class Segmentation(str, Enum):
        char = "char"
        word = "word"

    include_img_ids: bool = True
    separator: str = " "
    join_string: Optional[str] = " "
    use_symbols: bool = True
    convert_spaces: bool = False
    input_space: str = "<space>"
    output_space: str = " "
    segmentation: Optional[Segmentation] = None
    print_confidence_scores: bool = False


@dataclass
class NetoutArgs:
    """Netout arguments

    Args:
        output_transform: Apply this transformation at the end of the model.
            For instance, use "softmax" to get posterior probabilities as the output
            of the model
        matrix: Path of the Kaldi's archive containing the output matrices
            (one for each sample), where each row represents a timestep and
            each column represents a CTC label
        lattice: Path of the Kaldi's archive containing the output lattices
            (one for each sample), representing the CTC output
        digits: Number of digits to be used for formatting
    """

    class OutputTransform(str, Enum):
        softmax = "softmax"
        log_softmax = "log_softmax"

    output_transform: Optional[OutputTransform] = None
    matrix: Optional[str] = None
    lattice: Optional[str] = None
    digits: NonNegativeInt = 10
