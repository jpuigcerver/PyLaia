# Usage

Before using PyLaia, you need to format your dataset in a specific format. See [this page](./datasets/index.md).

Once the dataset is formatted and `pylaia` is installed and in your environment, you may use the following commands:

`pylaia-htr-create-model`
: To create a new PyLaia model. More details in the [dedicated page](./initialization/index.md).

`pylaia-htr-train-ctc`
: To train a PyLaia model. More details in the [dedicated page](./training/index.md).

`pylaia-htr-decode-ctc`
: To predict using a trained PyLaia model. More details in the [dedicated page](./prediction/index.md).

`pylaia-htr-netout`
: To dump features from a PyLaia model. Under construction.


PyLaia can also be used in combination with a n-gram language model to improve recognition. More details in the [dedicated page](./language_models/index.md).
