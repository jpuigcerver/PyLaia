# Explicit language modeling with n-grams

PyLaia supports lattice rescoring using a statistical language model.
This documentation gives instructions to build a language model with [kenlm](https://kheafield.com/code/kenlm/).

!!! note
    You can also use [SRILM](http://www.speech.sri.com/projects/srilm/) to build an ARPA language model.

To decode with a language model, you need:

* [a language model](./index.md#build-the-language-model)
* [a list of tokens](./index.md#list-of-tokens)
* [a lexicon](./index.md#lexicon)

## Build the language model

### Install kenlm

To build the language model, you first need to install and compile [kenlm](https://github.com/kpu/kenlm) by following the instructions detailed in the [README](https://github.com/kpu/kenlm#compiling).

### Generate resources to train the language model

To train a language model, you need to generate a corpus containing the training text tokenized at character, subword or word level.

#### Characters

Here is a sample of text tokenized at character-level (`corpus_characters.txt`).
```text title="corpus_characters.txt"
u d e <space> i <space> r e s t a u r a n t e r ,
v æ r e t <space> u h y r e <space> m e g e t <space> s a m m e n , <space> o f t e <space> t i l <space> m a a l t i d e r <space> o g <space> t i l <space> t h e <space> h o s <space> O s s b a h r ,
v i <space> s i d d e r <space> v e d <space> k a m i n e n <space> d e r <space> o g <space> s n a k k e r , <space> h v i l k e t <space> e r <space> m e g e t <space> m o r s o m t . <space> N u
k o m m e r <space> d e r <space> m a n g e <space> r e i s e n d e <space> v e n n e r <space> e l l e r <space> s l æ g t <space> e l l e r <space> p r i n s e s s e r , <space> s o m
O s s b a h r <space> m a a <space> v æ r e <space> s a m m e n <space> m e d <space> H e d b e r g <space> o f t e <space> o g s a a . <space> M e n <space> v i <space> k a n <space> l e v e
```

#### Subwords

Here is a sample of text tokenized at subword-level (`corpus_subwords.txt`).
```text title="corpus_subwords.txt"
ud e <space> i <space> r e st au r ant er ,
været <space> u h y r e <space> meget <space> sammen , <space> ofte <space> til <space> ma altid er <space> og <space> til <space> th e <space> hos <space> O s s ba h r ,
vi <space> sidde r <space> ved <space> ka min en <space> der <space> og <space> snakke r , <space> hvilket <space> er <space> meget <space> morsomt . <space> Nu
kommer <space> der <space> mange <space> r e i sende <space> venner <space> eller <space> s læg t <space> eller <space> pr in s e s ser , <space> som
O s s ba h r <space> maa <space> være <space> sammen <space> med <space> H e d berg <space> ofte <space> ogsaa . <space> Men <space> vi <space> kan <space> lev e
```

#### Words
Here is a sample of text tokenized at word-level (`corpus_words.txt`).
```text title="corpus_words.txt"
ude <space> i <space> restauranter <space> ,
været <space> uhyre <space> meget <space> sammen <space> , <space> ofte <space> til <space> maaltider <space> og <space> til <space> the <space> hos <space> Ossbahr <space> ,
vi <space> sidder <space> ved <space> kaminen <space> der <space> og <space> snakker <space> , <space> hvilket <space> er <space> meget <space> morsomt <space> . <space> Nu
kommer <space> der <space> mange <space> reisende <space> venner <space> eller <space> slægt <space> eller <space> prinsesser <space> , <space> som
Ossbahr <space> maa <space> være <space> sammen <space> med <space> Hedberg <space> ofte <space> ogsaa <space> . <space> Men <space> vi <space> kan <space> leve
```

### Train the language model

Once your corpus is created, you can estimate the n-gram model.

#### Characters

At character-level, we recommend building a 6-gram model. Use the following command:

```sh
bin/lmplz --order 6 \
    --text my_dataset/language_model/corpus_characters.txt \
    --arpa my_dataset/language_model/model_characters.arpa \
    --discount_fallback
```

!!! note
    The `--discount_fallback` option can be removed if your corpus is very large.

The following message should be displayed if the language model was built successfully:

```sh
=== 1/5 Counting and sorting n-grams ===
Reading language_model/corpus.txt
----5---10---15---20---25---30---35---40---45---50---55---60---65---70---75---80---85---90---95--100
****************************************************************************************************
Unigram tokens 111629 types 109
=== 2/5 Calculating and sorting adjusted counts ===
Chain sizes: 1:1308 2:784852864 3:1471599104 4:2354558464 5:3433731328 6:4709116928
Statistics:
1 109 D1=0.586207 D2=0.534483 D3+=1.5931
2 1734 D1=0.538462 D2=1.09853 D3+=1.381
3 7957 D1=0.641102 D2=1.02894 D3+=1.37957
4 17189 D1=0.747894 D2=1.20483 D3+=1.41084
5 25640 D1=0.812458 D2=1.2726 D3+=1.57601
6 32153 D1=0.727411 D2=1.13511 D3+=1.42722
Memory estimate for binary LM:
type      kB
probing 1798 assuming -p 1.5
probing 2107 assuming -r models -p 1.5
trie     696 without quantization
trie     313 assuming -q 8 -b 8 quantization
trie     648 assuming -a 22 array pointer compression
trie     266 assuming -a 22 -q 8 -b 8 array pointer compression and quantization
=== 3/5 Calculating and sorting initial probabilities ===
Chain sizes: 1:1308 2:27744 3:159140 4:412536 5:717920 6:1028896
----5---10---15---20---25---30---35---40---45---50---55---60---65---70---75---80---85---90---95--100
####################################################################################################
=== 4/5 Calculating and writing order-interpolated probabilities ===
Chain sizes: 1:1308 2:27744 3:159140 4:412536 5:717920 6:1028896
----5---10---15---20---25---30---35---40---45---50---55---60---65---70---75---80---85---90---95--100
####################################################################################################
=== 5/5 Writing ARPA model ===
----5---10---15---20---25---30---35---40---45---50---55---60---65---70---75---80---85---90---95--100
****************************************************************************************************
Name:lmplz	VmPeak:12643224 kB	VmRSS:6344 kB	RSSMax:1969316 kB	user:0.196445	sys:0.514686	CPU:0.711161	real:0.682693
```

#### Subwords

At subword-level, we recommend building a 6-gram model. Use the following command:

```sh
bin/lmplz --order 6 \
    --text my_dataset/language_model/corpus_subwords.txt \
    --arpa my_dataset/language_model/model_subwords.arpa \
    --discount_fallback
```

!!! note
    The `--discount_fallback` option can be removed if your corpus is very large.

#### Words

At word-level, we recommend building a 3-gram model. Use the following command:

```sh
bin/lmplz --order 3 \
    --text my_dataset/language_model/corpus_words.txt \
    --arpa my_dataset/language_model/model_words.arpa \
    --discount_fallback
```

!!! note
    The `--discount_fallback` option can be removed if your corpus is very large.

## Predict with a language model

Once the language model is trained, you need to generate a list of tokens and a lexicon.

### List of tokens

The list of tokens `tokens.txt` lists all the tokens that can be predicted by PyLaia.
It should be similar to `syms.txt`, but without any index, and can be generated with this command:
```bash
cut -d' ' -f 1 syms.txt > tokens.txt
```

!!! note
    This file does not depend on the tokenization level.

```text title="tokens.txt"
<ctc>
.
,
a
b
c
...
<space>
```

### Lexicon

The lexicon lists all the words in the vocabulary and its decomposition in tokens.

#### Characters

At character-level, words are simply characters, so the `lexicon_characters.txt` file should map characters to characters:

```text title="lexicon_characters.txt"
<ctc> <ctc>
. .
, ,
a a
b b
c c
...
<space> <space>
```

#### Subwords
At subword-level, the `lexicon_subwords.txt` file should map subwords with their character decomposition:

```text title="lexicon_subwords.txt"
<ctc> <ctc>
. .
, ,
altid a l t i d
ant a n t
au a u
...
<space> <space>
```

#### Words
At word-level, the `lexicon_words.txt` file should map words with their character decomposition:

```text title="lexicon_words.txt"
<ctc> <ctc>
. .
, ,
der d e r
er e r
eller e l l e r
...
<space> <space>
```

### Predict with PyLaia

See the [dedicated example](../prediction/index.md#predict-with-a-language-model).
