import torch
import pytest
from pathlib import Path
from laia.decoders import CTCLanguageDecoder

tokens = """<ctc>
a
e
h
i
s
t
.
<unk>
<space>"""

lexicon="""<ctc> <ctc>
a a
e e
h h
i i
s s
t t
. .
<unk> <unk>
<space> <space>"""

arpa_lm="""\\data\\
ngram 1=10
ngram 2=14

\\1-grams:
-1.09691\t.\t-0.2648178
-1.09691\t</s>
-99\t<s>\t-0.2253093
-0.79588\t<space>\t-0.10721
-1.09691\ta\t-0.2253093
-1.09691\te\t-0.2253093
-1.09691\th\t-0.2455126
-0.9208187\ti\t-0.4014006
-0.79588\ts\t-0.2304489
-0.79588\tt\t-0.1818436

\\2-grams:
-0.30103\t. </s>
-0.30103\t<s> t
-0.7781513\t<space> a
-0.7781513\t<space> i
-0.7781513\t<space> t
-0.30103\ta <space>
-0.30103\te s
-0.30103\th i
-0.1760913\ti s
-0.39794\ts <space>
-0.69897\ts t
-0.7781513\tt .
-0.7781513\tt e
-0.7781513\tt h

\\end\\"""


@pytest.mark.parametrize(
    ["input_tensor", "lm_weight", "expected_result"],
    [
        (torch.tensor([
            [[-2.1, -1.3, -4.1, -4.2, -5.0, -5.1, -0.2, -4.2, -5.2, -1.2]],
            [[-2.1, -0.1, -0.3, -4.2, -5.0, -5.1, -4.7, -4.2, -5.2, -1.2]],
            [[-2.1, -2.5, -4.1, -4.2, -5.0, -0.7, -0.9, -1.7, -5.2, -1.2]],
            [[-2.1, -0.5, -4.1, -4.2, -5.0, -0.7, -0.1, -1.1, -5.2, -1.2]],
        ]), 
        0, 
        "tast"
    ), (torch.tensor(
        [
            [[-2.1, -1.3, -4.1, -4.2, -5.0, -5.1, -0.2, -4.2, -5.2, -1.2]],
            [[-2.1, -0.1, -0.3, -4.2, -5.0, -5.1, -4.7, -4.2, -5.2, -1.2]],
            [[-2.1, -2.5, -4.1, -4.2, -5.0, -0.7, -0.9, -1.7, -5.2, -1.2]],
            [[-2.1, -0.5, -4.1, -4.2, -5.0, -0.7, -0.1, -1.1, -5.2, -1.2]],
        ]
    ), 1, "test")
    ]
)
def test_lm_decoding_weight(tmpdir, input_tensor, lm_weight, expected_result):
    tokens_path = Path(tmpdir) / "tokens.txt"
    lexicon_path = Path(tmpdir) / "lexicon.txt"
    arpa_path = Path(tmpdir) / "lm.arpa"
    tokens_path.write_text(tokens, "utf-8")
    lexicon_path.write_text(lexicon, "utf-8")
    arpa_path.write_bytes(bytes(arpa_lm, "utf-8"))

    decoder = CTCLanguageDecoder(
        language_model_path=str(arpa_path), 
        tokens_path=str(tokens_path), 
        lexicon_path=str(lexicon_path),
        language_model_weight=lm_weight
        )

    with open(tokens_path, "r") as f:
        tokens_char = f.read().splitlines()

    r = decoder(input_tensor)
    expected_result_index = [tokens_char.index(char) for char in expected_result]
    assert r["hyp"][0]==expected_result_index
