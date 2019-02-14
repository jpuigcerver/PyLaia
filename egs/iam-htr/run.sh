#!/bin/bash
set -e;

# Directory where the run.sh script is placed.
SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)";
cd "$SDIR";

iam_username="$IAM_USERNAME";
iam_password="$IAM_PASSWORD";
help_message="
Usage: ${0##*/} [options]

Options:
  --iam_password : (type = string, default = \"$iam_password\")
                   Password for the IAM server.
  --iam_username : (type = string, default = \"$iam_username\")
                   Username for the IAM server.
";
source ../utils/parse_options.inc.sh || exit 1;


export PATH="$PWD/../..:$PATH";

# Step 1. Download data.
./src/download.sh \
  --iam_username "$iam_username" \
  --iam_password  "$iam_password";

# Step 2. Prepare images.
./src/prepare_images.sh;

# Step 3. Prepare text.
./src/prepare_texts.sh;

# Step 4. Train the neural network.
./src/train_puigcerver17.sh;

# Step 5. Decode using only the neural network.
#./src/decode_net.sh;
