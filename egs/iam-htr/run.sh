#!/bin/bash
set -e;

# Directory where the run.sh script is placed.
SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)";
[ "$(pwd)" != "$SDIR" ] &&
echo "Please, run this script from the experiment top directory!" >&2 &&
exit 1;

# Step 1. Download data.
./src/download.sh --iam_user "$IAM_USER" --iam_pass "$IAM_PASS";

# Step 2. Prepare images.
./src/prepare_images.sh;

# Step 3. Prepare text.
./src/prepare_iam_text.sh;

# Step 4. Train the neural network.
./src/train.sh;

# Step 5. Decode using only the neural network.
./src/decode_net.sh;
