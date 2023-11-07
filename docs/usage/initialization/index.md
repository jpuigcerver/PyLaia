# Model initialization

The `pylaia-htr-create-model` command can be used to create a PyLaia model. To know more about the options of this command, use `pylaia-htr-create-model --help`.

## Purpose

The general architecture of PyLaia is composed of convolutional blocks followed by a set a bi-directionnal recurrent layers and a linear layer. PyLaia is fully configurable by the user, including:

- Number of convolutional blocks,
- Number of recurrent layers,
- Batch normalization,
- Pooling layers,
- Activation function,
- ...

This command will create a pickled file (named `model` by default), which is required to initialize the `LaiaCRNN` class before training.

## Parameters

The full list of parameters is detailed in this section.


### General parameters

| Parameter            | Description                                                                                                                                                                                                | Type   | Default      |
| -------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------ | ------------ |
| `syms`               | Positional argument. Path to a file mapping characters to integers. The CTC symbol must be mapped to integer 0.                                                                                            | `str`  |              |
| `config`             | Path to a JSON configuration file                                                                                                                                                                          | `json` |              |
| `fixed_input_height` | Height of the input images. If set to 0, a variable height model will be used (see `adaptive_pooling`). This will be used to compute the model output height at the end of the convolutional layers.       | `int`  | 0            |
| `adaptive_pooling`   | Use custom adaptive pooling layers to enable training with variable height images. Takes into account the size of each individual image within the batch (before padding). Should be in `{avg,max}pool-N`. | `str`  | `avgpool-16` |
| `save_model`         | Whether to save the model to a file.                                                                                                                                                                       | `bool` | `True`       |

### Common parameters

| Name                    | Description                             | Type  | Default |
| ----------------------- | --------------------------------------- | ----- | ------- |
| `common.train_path`     | Directory where the model will be saved | `str` | `.`     |
| `common.model_filename` | Filename of the model.                  | `str` | `model` |

### Logging arguments

| Name                      | Description                                                                                                    | Type            | Default                                           |
| ------------------------- | -------------------------------------------------------------------------------------------------------------- | --------------- | ------------------------------------------------- |
| `logging.fmt`             | Logging format.                                                                                                | `str`           | `%(asctime)s %(levelname)s %(name)s] %(message)s` |
| `logging.level`           | Logging level. Should be in `{NOTSET,DEBUG,INFO,WARNING,ERROR,CRITICAL}`                                       | `Level`         | `INFO`                                            |
| `logging.filepath`        | Filepath for the logs file. Can be a filepath or a filename to be created in `train_path`/`experiment_dirname` | `Optional[str]` |                                                   |
| `logging.overwrite`       | Whether to overwrite the logfile or to append.                                                                 | `bool`          | `False`                                           |
| `logging.to_stderr_level` | If filename is set, use this to log also to stderr at the given level.                                         | `Level`         | `ERROR`                                           |

### Architecture arguments


| Name                      | Description                                                                                         | Type    | Default                                                |
| ------------------------- | --------------------------------------------------------------------------------------------------- | ------- | ------------------------------------------------------ |
| `crnn.num_input_channels` | Number of channels of the input images.                                                             | `int`   | `1`                                                    |
| `crnn.vertical_text`      | Whether the text is written vertically.                                                             | `bool`  | `False`                                                |
| `crnn.cnn_num_features`   | Number of features in each convolutional layer.                                                     | `List`  | `[16, 16, 32, 32]`                                     |
| `crnn.cnn_kernel_size`    | Kernel size of each convolutional layer (e.g. [n,n,...] or [[h1,w1],[h2,w2],...]).                  | `List`  | `[3, 3, 3, 3]`                                         |
| `crnn.cnn_stride`         | Stride of each convolutional layer. (e.g. [n,n,...] or [[h1,w1],[h2,w2],...])                       | `List`  | `[1, 1, 1, 1]`                                         |
| `crnn.cnn_dilation`       | Spacing between each convolutional layer kernel elements. (e.g. [n,n,...] or [[h1,w1],[h2,w2],...]) | `List`  | `[1, 1, 1, 1]`                                         |
| `crnn.cnn_activation`     | Type of activation function in each convolutional layer (from `torch.nn`).                          | `List`  | `['LeakyReLU', 'LeakyReLU', 'LeakyReLU', 'LeakyReLU']` |
| `crnn.cnn_poolsize`       | MaxPooling size after each convolutional layer. (e.g. [n,n,...] or [[h1,w1],[h2,w2],...]).          | `List`  | `[2, 2, 2, 0]`                                         |
| `crnn.cnn_dropout`        | Dropout probability at the input of each convolutional layer.                                       | `List`  | `[0.0, 0.0, 0.0, 0.0]`                                 |
| `crnn.cnn_batchnorm`      | Whether to do batch normalization before the activation in each convolutional layer.                | `List`  | `[False, False, False, False]`                         |
| `crnn.use_masks`          | Whether to apply a zero mask after each convolution and non-linear activation.                      | `bool`  | `False`                                                |
| `crnn.rnn_layers`         | Number of recurrent layers.                                                                         | `int`   | `3`                                                    |
| `crnn.rnn_units`          | Number of units in each recurrent layer.                                                            | `int`   | `256`                                                  |
| `crnn.rnn_dropout`        | Dropout probability at the input of each recurrent layer.                                           | `float` | `0.5`                                                  |
| `crnn.rnn_type`           | Type of recurrent layer (from `torch.nn`).                                                          | `str`   | `LSTM`                                                 |
| `crnn.lin_dropout`        | Dropout probability at the input of the final linear layer.                                         | `float` | `0.5`                                                  |

## Examples

The model can be configured using command-line arguments or a YAML configuration file. Note that CLI arguments override the values from the configuration file.


### Example with Command Line Arguments (CLI)

Run the following command to create a model:
```sh
pylaia-htr-create-model /path/to/syms.txt \
   --fixed_input_height 128 \
   --crnn.rnn_layers 4 \
   --logging.filepath model.log \
   --common.train_path my_experiments/
```

### Example with a YAML configuration file

Run the following command to create a model:
```sh
pylaia-htr-create-model --config config_create_model.yaml
```

Where `config_create_model.yaml` is:

```yaml
crnn:
  cnn_activation:
  - LeakyReLU
  - LeakyReLU
  - LeakyReLU
  - LeakyReLU
  cnn_batchnorm:
  - true
  - true
  - true
  - true
  cnn_dilation:
  - 1
  - 1
  - 1
  - 1
  cnn_kernel_size:
  - 3
  - 3
  - 3
  - 3
  cnn_num_features:
  - 12
  - 24
  - 48
  - 48
  cnn_poolsize:
  - 2
  - 2
  - 0
  - 2
  lin_dropout: 0.5
  rnn_dropout: 0.5
  rnn_layers: 3
  rnn_type: LSTM
  rnn_units: 256
fixed_input_height: 128
save_model: true
syms: /path/to/syms.txt
```
