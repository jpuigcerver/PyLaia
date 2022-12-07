from typing import Any, Dict, List

import numpy as np
import torch
from torchaudio.models.decoder import ctc_decoder

from laia.losses.ctc_loss import transform_batch


class CTCLanguageDecoder:
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
        x: Any,
    ) -> Dict[str, List]:
        x, xs = transform_batch(x)
        x = x.detach()

        # no GPU support
        device = torch.device("cpu")

        # from (frame, bs, num_tokens) to (bs, frame, num_tokens)
        x = x.permute((1, 0, 2))
        x = torch.nn.functional.log_softmax(x, dim=-1)
        x = x.to(device)
        if isinstance(xs, list):
            xs = torch.tensor(xs)
            xs.to(device)

        # decode
        hypotheses = self.decoder(x, xs)
        out = {}
        out["hyp"] = [hypothesis[0].tokens.tolist()[1:-1] for hypothesis in hypotheses]
        # no character-based probability
        # however, a score can be accessed with hypothesis[0].score
        return out
