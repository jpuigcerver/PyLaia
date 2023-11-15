# Decoding

The `pylaia-htr-decode-ctc` command can be used to predict using a trained PyLaia model. To know more about the options of this command, use `pylaia-htr-decode-ctc --help`.

## Purpose

This command uses a trained PyLaia model to predict on a dataset.

It requires:

- a [list of image ids](../datasets/index.md#image-names),
- the pickled `model` file created during [model initialization](../initialization/index.md),
- the weights `*.ckpt` of the trained model created during [model training](../training/index.md).

## Parameters

The full list of parameters is detailed in this section.

### General parameters

| Parameter  | Description                                                                                                         | Type   | Default |
| ---------- | ------------------------------------------------------------------------------------------------------------------- | ------ | ------- |
| `syms`     | Positional argument. Path to a file mapping characters to integers. The CTC symbol **must** be mapped to integer 0. | `str`  |         |
| `img_list` | Positional argument. File containing the names of the images to decode (one image per line).                        | `str`  |         |
| `img_dirs` | Directories containing line images.                                                                                 | `str`  |         |
| `config`   | Path to a JSON configuration file                                                                                   | `json` |         |

### Common parameters

| Name                        | Description                                                                                                                                                                                                                                         | Type         | Default |
| --------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------ | ------- |
| `common.train_path`         | Directory where the model will be saved                                                                                                                                                                                                             | `str`        | `.`     |
| `common.model_filename`     | Filename of the model.                                                                                                                                                                                                                              | `str`        | `model` |
| `common.experiment_dirname` | Directory name of the experiment.                                                                                                                                                                                                                   | `experiment` | `74565` |
| `common.checkpoint`         | Checkpoint to load. Must be a filepath, a filename, a glob pattern or `None` (in this case, the best checkpoint will be loaded). Note that the checkpoint will be searched in `common.experiment_dirname`, unless you provide an absolute filepath. | `int`        | `None`  |

### Data arguments

| Name              | Description                                      | Type        | Default       |
| ----------------- | ------------------------------------------------ | ----------- | ------------- |
| `data.batch_size` | Batch size.                                      | `int`       | `8`           |
| `data.color_mode` | Color mode. Must be either `L`, `RGB` or `RGBA`. | `ColorMode` | `ColorMode.L` |

### Decode arguments

| Name                                  | Description                                                                                                                                                                         | Type            | Default   |
| ------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------- | --------- |
| `decode.include_img_ids`              | Include the associated image ids in the decoding/segmentation output                                                                                                                | `bool`          | `True`    |
| `decode.separator`                    | String to use as a separator between the image ids and the decoding/segmentation output.                                                                                            | `str`           | ` `       |
| `decode.join_string`                  | String to use to join the decoding output.                                                                                                                                          | `Optional[str]` | ` `       |
| `decode.use_symbols`                  | Convert the decoding output to symbols instead of symbol index.                                                                                                                     | `bool`          | `True`    |
| `decode.convert_spaces`               | Whether or not to convert spaces.                                                                                                                                                   | `bool`          | `False`   |
| `decode.input_space`                  | Replace the space by this symbol if `convert_spaces` is set. Used for word segmentation and confidence score computation.                                                           | `str`           | `<space>` |
| `decode.output_space`                 | Space symbol to display during decoding.                                                                                                                                            | `str`           | ` `       |
| `decode.segmentation`                 | Use CTC alignment to estimate character or word segmentation. Should be `char` or `word`.                                                                                           | `Optional[str]` | `None `   |
| `decode.temperature`                  | Temperature parameters used to scale the logits.                                                                                                                                    | `float`         | `1.0`     |
| `decode.print_line_confidence_scores` | Whether to print line confidence scores.                                                                                                                                            | `bool`          | `False`   |
| `decode.print_line_confidence_scores` | Whether to print word confidence scores.                                                                                                                                            | `bool`          | `False`   |
| `decode.use_language_model`           | Whether to decode with an external language model.                                                                                                                                  | `bool`          | `False`   |
| `decode.language_model_path`          | Path to a KenLM or ARPA n-gram language model.                                                                                                                                      | `str`           | `None`    |
| `decode.language_model_weight`        | Weight of the language model.                                                                                                                                                       | `float`         | `None`    |
| `decode.tokens_path`                  | Path to a file containing valid tokens. If using a file, the expected format is for tokens mapping to the same index to be on the same line. The `ctc` symbol should be at index 0. | `str`           | `None`    |
| `decode.lexicon_path`                 | Path to a lexicon file containing the possible words and corresponding spellings.                                                                                                   | `str`           | `None`    |
| `decode.unk_token`                    | String representing unknown characters.                                                                                                                                             | `str`           | `<unk>`   |
| `decode.blank_token`                  | String representing the blank/ctc symbol.                                                                                                                                           | `str`           | `<ctc>`   |


### Logging arguments

| Name                      | Description                                                                                                    | Type            | Default                                           |
| ------------------------- | -------------------------------------------------------------------------------------------------------------- | --------------- | ------------------------------------------------- |
| `logging.fmt`             | Logging format.                                                                                                | `str`           | `%(asctime)s %(levelname)s %(name)s] %(message)s` |
| `logging.level`           | Logging level. Should be in `{NOTSET,DEBUG,INFO,WARNING,ERROR,CRITICAL}`                                       | `Level`         | `INFO`                                            |
| `logging.filepath`        | Filepath for the logs file. Can be a filepath or a filename to be created in `train_path`/`experiment_dirname` | `Optional[str]` |                                                   |
| `logging.overwrite`       | Whether to overwrite the logfile or to append.                                                                 | `bool`          | `False`                                           |
| `logging.to_stderr_level` | If filename is set, use this to log also to stderr at the given level.                                         | `Level`         | `ERROR`                                           |

### Trainer arguments

Pytorch Lightning `Trainer` flags can also be set using the `--trainer` argument. See [the documentation](https://github.com/Lightning-AI/lightning/blob/1.7.0/docs/source-pytorch/common/trainer.rst#trainer-flags).

This flag is mostly useful to define whether to predict on CPU or GPU.

* `--trainer.gpus 0` to run on CPU,
* `--trainer.gpus n` to run on `n` GPUs (use with `--training.auto_select True` for auto-selection),
* `--trainer.gpus -1` to run on all GPUs.


## Examples

The prediction can be done using command-line arguments or a YAML configuration file. Note that CLI arguments override the values from the configuration file.

### Predict using a model from Hugging Face

First, clone a trained model from Hugging Face:
```bash
git clone https://huggingface.co/Teklia/pylaia-huginmunin
```

List image names in `img_list.txt`:
```text
docs/assets/219007024-f45433e7-99fd-43b0-bce6-93f63fa72a8f
docs/assets/219008758-c0097bb4-c55a-4652-ad2e-bba350bee0e4
```

Predict with:
```bash
pylaia-htr-decode-ctc --common.experiment_dirname pylaia-huginmunin/ \
                      --common.model_filename pylaia-huginmunin/model \
                      --img_dir [docs/assets] \
                      pylaia-huginmunin/syms.txt \
                      img_list.txt
```

Expected output:
```text
219007024-f45433e7-99fd-43b0-bce6-93f63fa72a8f o g <space> V a l s t a d <space> k a n <space> v i <space> v i s t
219008758-c0097bb4-c55a-4652-ad2e-bba350bee0e4 i k k e <space> g j ø r e <space> R e g n i n g <space> p a a ,
```

Note that by default, each token is separated by a space, and the space symbol is represented by `--decode.input_space` (default: `"<space>"`).

### Predict with a YAML configuration file

Run the following command to predict a model on CPU using:
```bash
pylaia-htr-decode-ctc --config config_decode_model.yaml
```

With the following configuration file:
```yaml title="config_decode_model.yaml"
syms: pylaia-huginmunin/syms.txt
img_list: img_list.txt
img_dirs:
  - docs/assets/
common:
  experiment_dirname: pylaia-huginmunin
  model_filename: pylaia-huginmunin/model
decode:
  join_string: ""
  convert_spaces: true
trainer:
  gpus: 0
```

Expected output:
```text
219007024-f45433e7-99fd-43b0-bce6-93f63fa72a8f og Valstad kan vi vist
219008758-c0097bb4-c55a-4652-ad2e-bba350bee0e4 ikke gjøre Regning paa,
```

Note that setting `--decode.join_string ""` and `--decode.convert_spaces True` will display the text well formatted.

### Predict with confidence scores

PyLaia estimate character probability for each timestep. It is possible to print the probability at line or word level.

#### Line confidence scores

Run the following command to predict with line confidence scores:
```bash
pylaia-htr-decode-ctc --config config_decode_model.yaml \
                      --decode.print_line_confidence_score True
```

Expected output:
```text
219007024-f45433e7-99fd-43b0-bce6-93f63fa72a8f 0.99 og Valstad kan vi vist
219008758-c0097bb4-c55a-4652-ad2e-bba350bee0e4 0.98 ikke gjøre Regning paa,
```

#### Word confidence scores

Run the following command to predict with word confidence scores:
```bash
pylaia-htr-decode-ctc --config config_decode_model.yaml \
                      --decode.print_word_confidence_score True
```

Expected output:
```text
219007024-f45433e7-99fd-43b0-bce6-93f63fa72a8f ['1.00', '1.00', '1.00', '1.00', '1.00'] og Valstad kan vi vist
219008758-c0097bb4-c55a-4652-ad2e-bba350bee0e4 ['1.00', '0.91', '1.00', '0.99'] ikke gjøre Regning paa,
```

#### Temperature scaling

PyLaia tends to output overly confident probabilities. [Temperature scaling](https://arxiv.org/pdf/1706.04599.pdf) can be used to improve the reliability of confidence scores. The best temperature can be determined with a grid search algorithm by maximizing the correlation between 1-CER and confidence scores.

Run the following command to predict callibrated word confidence scores with `temperature=3.0`
```bash
pylaia-htr-decode-ctc --config config_decode_model.yaml \
                      --decode.print_word_confidence_score True \
                      --decode.temperature 3.0
```

Expected output:
```text
219007024-f45433e7-99fd-43b0-bce6-93f63fa72a8f ['0.93', '0.85', '0.87', '0.93', '0.85'] og Valstad kan vi vist
219008758-c0097bb4-c55a-4652-ad2e-bba350bee0e4 ['0.93', '0.84', '0.86', '0.83'] ikke gjøre Regning paa,
```

### Predict with a language model

PyLaia supports KenLM and ARPA language models.

Once the n-gram model is built, run the following command to combine it to your PyLaia model:
```bash
pylaia-htr-decode-ctc --config config_decode_model_lm.yaml
```

With the following configuration file:
```yaml title="config_decode_model_lm.yaml"
syms: pylaia-huginmunin/syms.txt
img_list: img_list.txt
img_dirs:
  - docs/assets/
common:
  experiment_dirname: pylaia-huginmunin
  model_filename: pylaia-huginmunin/model
decode:
  join_string: ""
  convert_spaces: true
  use_language_model: true
  language_model_path: pylaia-huginmunin/language_model.arpa.gz
  tokens_path: pylaia-huginmunin/tokens.txt
  lexicon_path: pylaia-huginmunin/lexicon.txt
  language_model_weight: 1.5
  decode.print_line_confidence_score: true
trainer:
  gpus: 0
```

Expected output:
```text
219007024-f45433e7-99fd-43b0-bce6-93f63fa72a8f 0.90 og Valstad kan vi vist
219008758-c0097bb4-c55a-4652-ad2e-bba350bee0e4 0.89 ikke gjøre Regning paa,
```

### Predict with CTC alignement

It is possible to estimate text localization based on CTC alignments with the `--decode.segmentation` option. It returns a list texts with their estimated coordinates: `(text, x1, y1, x2, y2)`.

#### Character level

To output character localization, use the `--decode.segmentation char` option:
```bash
pylaia-htr-decode-ctc --common.experiment_dirname pylaia-huginmunin/ \
                      --common.model_filename pylaia-huginmunin/model \
                      --decode.segmentation char \
                      --img_dir [docs/assets] \
                      pylaia-huginmunin/syms.txt \
                      img_list.txt
```

Expected output:
```text
219007024-f45433e7-99fd-43b0-bce6-93f63fa72a8f [('o', 1, 1, 31, 128), ('g', 32, 1, 79, 128), ('<space>', 80, 1, 143, 128), ('V', 144, 1, 167, 128), ('a', 168, 1, 223, 128), ('l', 224, 1, 255, 128), ('s', 256, 1, 279, 128), ('t', 280, 1, 327, 128), ('a', 328, 1, 367, 128), ('d', 368, 1, 407, 128), ('<space>', 408, 1, 496, 128), ('k', 497, 1, 512, 128), ('a', 513, 1, 576, 128), ('n', 577, 1, 624, 128), ('<space>', 625, 1, 712, 128), ('v', 713, 1, 728, 128), ('i', 729, 1, 776, 128), ('<space>', 777, 1, 808, 128), ('v', 809, 1, 824, 128), ('i', 825, 1, 872, 128), ('s', 873, 1, 912, 128), ('t', 913, 1, 944, 128)]
219008758-c0097bb4-c55a-4652-ad2e-bba350bee0e4 [('i', 1, 1, 23, 128), ('k', 24, 1, 71, 128), ('k', 72, 1, 135, 128), ('e', 136, 1, 191, 128), ('<space>', 192, 1, 248, 128), ('g', 249, 1, 264, 128), ('j', 265, 1, 312, 128), ('ø', 313, 1, 336, 128), ('r', 337, 1, 376, 128), ('e', 377, 1, 408, 128), ('<space>', 409, 1, 481, 128), ('R', 482, 1, 497, 128), ('e', 498, 1, 545, 128), ('g', 546, 1, 569, 128), ('n', 570, 1, 601, 128), ('i', 602, 1, 665, 128), ('n', 666, 1, 706, 128), ('g', 707, 1, 762, 128), ('<space>', 763, 1, 794, 128), ('p', 795, 1, 802, 128), ('a', 803, 1, 850, 128), ('a', 851, 1, 890, 128), (',', 891, 1, 914, 128)]
```

#### Word level

To output word localization, use the `--decode.segmentation word` option:
```bash
pylaia-htr-decode-ctc --common.experiment_dirname pylaia-huginmunin/ \
                      --common.model_filename pylaia-huginmunin/model \
                      --decode.segmentation word \
                      --img_dir [docs/assets] \
                      pylaia-huginmunin/syms.txt \
                      img_list.txt
```

Expected output:
```text
219007024-f45433e7-99fd-43b0-bce6-93f63fa72a8f [('og', 1, 1, 79, 128), ('<space>', 80, 1, 143, 128), ('Valstad', 144, 1, 407, 128), ('<space>', 408, 1, 496, 128), ('kan', 497, 1, 624, 128), ('<space>', 625, 1, 712, 128), ('vi', 713, 1, 776, 128), ('<space>', 777, 1, 808, 128), ('vist', 809, 1, 944, 128)]
219008758-c0097bb4-c55a-4652-ad2e-bba350bee0e4 [('ikke', 1, 1, 191, 128), ('<space>', 192, 1, 248, 128), ('gjøre', 249, 1, 408, 128), ('<space>', 409, 1, 481, 128), ('Regning', 482, 1, 762, 128), ('<space>', 763, 1, 794, 128), ('paa,', 795, 1, 914, 128)]
```
