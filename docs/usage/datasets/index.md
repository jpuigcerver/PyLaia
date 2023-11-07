# Dataset formatting

To train PyLaia, you need line images and their corresponding transcriptions. The dataset should be divided into three sets: training, validation and test sets.

The dataset should be formatted as follows:
```
# Images
├── images
    ├── train/
    ├── val/
    └── test/
# Tokenized transcriptions (used for training)
├── train.txt
├── val.txt
├── test.txt
# Image ids (used for prediction)
├── train_ids.txt
├── val_ids.txt
├── test_ids.txt
# Transcriptions (used for evaluation)
├── train_text.txt
├── val_text.txt
├── test_text.txt
# Symbol list
└── syms.txt
```

## Images

By default, images should be resized to a fixed height (recommended value: 128 pixels). This can be done using [ImageMagick's `mogrify`](https://imagemagick.org/script/mogrify.php) function:
```
mogrify -resize x128 images/*.jpg
```

Note that PyLaia can also support variable size images by setting `--fixed_input_height 0` during [model initialization](../initialization/index.md).

## Tokenized transcriptions

Three files `{train|val|test}.txt` are required to train the model. They should map image names and tokenized transcriptions.

Example:

```sh
train/im01 f o r <space> d e t <space> t i l f æ l d e <space> d e t <space> s k u l d e <space> l y k k e s <space> D i g
train/im02 a t <space> o p d r i v e <space> d e t <space> o m s k r e v n e <space> e x p l : <space> a f
train/im03 « F r u <space> I n g e r » , <space> a t <space> s e n d e <space> m i g <space> s a m m e
```

## Image names

Three additional files `{train|val|test}_ids.txt` are required to run predictions. They should list image names without transcriptions.

Example:

```sh
train/im01
train/im02
train/im03
```

## Transcriptions

Finally, three files `{train|val|test}_text.txt` are required to evaluate your models. They should map image names and non-tokenized transcriptions.

Example:

```sh
train/im01 for det tilfælde det skulde lykkes Dig
train/im02 at opdrive det omskrevne expl: af
train/im03 «Fru Inger», at sende mig samme
```

## List of symbols

Finally, a file named `syms.txt` is required, mapping tokens from the training set and their index, starting with the `<ctc>` token.

Example:

```
<ctc> 0
! 1
" 2
& 3
' 4
( 5
) 6
+ 7
, 8
- 9
. 10
/ 11
0 12
1 13
2 14
3 15
4 16
5 17
6 18
7 19
8 20
9 21
: 22
; 23
< 24
= 25
> 26
? 27
A 28
B 29
C 30
D 31
E 32
F 33
G 34
H 35
I 36
J 37
K 38
L 39
M 40
N 41
O 42
P 43
Q 44
R 45
S 46
T 47
U 48
V 49
W 50
X 51
Y 52
Z 53
[ 54
] 55
a 56
b 57
c 58
d 59
e 60
f 61
g 62
h 63
i 64
j 65
k 66
l 67
m 68
n 69
o 70
p 71
q 72
r 73
s 74
t 75
u 76
v 77
w 78
x 79
y 80
z 81
« 82
¬ 83
» 84
¼ 85
½ 86
Å 87
Æ 88
Ø 89
à 90
á 91
â 92
ä 93
å 94
æ 95
ç 96
è 97
é 98
ê 99
ö 100
ø 101
ù 102
û 103
ü 104
– 105
— 106
’ 107
„ 108
… 109
<unk> 110
<space> 111
```
