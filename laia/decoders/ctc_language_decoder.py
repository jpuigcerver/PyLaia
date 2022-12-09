from typing import Any, Dict, List

import numpy as np
import torch
from torchaudio.models.decoder import ctc_decoder

from laia.losses.ctc_loss import transform_batch


class CTCLanguageDecoder:
    """
    Intialize a CTC decoder with n-gram language modeling.
    Args:
        language_model_path (str): path to a KenLM or ARPA language model
        lexicon_path (str): path to a lexicon file containing the possible words and corresponding spellings.
            Each line consists of a word and its space separated spelling. If `None`, uses lexicon-free
            decoding.
        tokens_path (str): path to a file containing valid tokens. If using a file, the expected
            format is for tokens mapping to the same index to be on the same line
        language_model_weight (float): weight of the language model.
        blank_token (str): token representing the blank/ctc symbol
        unk_token (str): token representing unknown characters
        sil_token (str): token representing the space character
    """

    def __init__(
        self,
        language_model_path: str,
        lexicon_path: str,
        tokens_path: str,
        language_model_weight: float = 1.0,
        blank_token: str = "<ctc>",
        unk_token: str = "<unk>",
        sil_token: str = "<space>",
    ):

        self.decoder = ctc_decoder(
            lm=language_model_path,
            lexicon=lexicon_path,
            tokens=tokens_path,
            lm_weight=language_model_weight,
            blank_token=blank_token,
            unk_word=unk_token,
            sil_token=sil_token,
        )

    def __call__(
        self,
        features: Any,
    ) -> Dict[str, List]:
        """
        Decode a feature vector using n-gram language modelling.
        Args:
            features (Any): feature vector of size (n_frame, batch_size, n_tokens).
                Can be either a torch.tensor or a torch.nn.utils.rnn.PackedSequence
        Returns:
            out (Dict[str, List]): a dictionnary containing the hypothesis (the list of decoded tokens).
                There is no character-based probability.
        """

        # Get the actual size of each feature in the batch
        batch_features, batch_sizes = transform_batch(features)
        batch_features = batch_features.detach()

        # Reshape from (n_frame, batch_size, n_tokens) to (batch_size, n_frame, n_tokens)
        batch_features = batch_features.permute((1, 0, 2))

        # Apply log softmax
        batch_features = torch.nn.functional.log_softmax(batch_features, dim=-1)

        # No GPU support for torchaudio's ctc_decoder
        device = torch.device("cpu")
        batch_features = batch_features.to(device)
        if isinstance(batch_sizes, list):
            batch_sizes = torch.tensor(batch_sizes)
            batch_sizes.to(device)

        # Decode
        hypotheses = self.decoder(batch_features, batch_sizes)

        # Format the output
        out = {}
        out["hyp"] = [hypothesis[0].tokens.tolist() for hypothesis in hypotheses]
        # you can get a log likelyhood with hypothesis[0].score
        return out
