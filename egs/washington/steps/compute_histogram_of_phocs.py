from __future__ import absolute_import
from __future__ import print_function

import argparse
from laia.utils.phoc import unigram_phoc
from future.utils import viewitems

parser = argparse.ArgumentParser()
parser.add_argument('input_text', type=argparse.FileType('r'))
parser.add_argument('phoc_levels', type=int)
args = parser.parse_args()

# Load George Washington vocabulary
vocabulary = set()
for line in args.input_text:
    word = line.split()[1]
    vocabulary.add(word)

# Recover George Washington alphabet
alphabet = set()
for word in vocabulary:
    alphabet.update([ch for ch in word])
alphabet = sorted(list(alphabet))

# Obtain the different PHOCs and count how many words produce the
# same PHOC code.
unigram_map = { c: i for i, c in enumerate(alphabet) }
phoc_counter = {}
for word in vocabulary:
    phoc = unigram_phoc(word, unigram_map, list(range(1, args.phoc_levels + 1)))
    if phoc in phoc_counter:
        phoc_counter[phoc] += 1
    else:
        phoc_counter[phoc] = 1

# Compute PHOC histogram
phoc_histogram = {}
for phoc, counter in viewitems(phoc_counter):
    c = phoc_histogram.get(counter, 0)
    phoc_histogram[counter] = c + 1

# Print histogram
for k, v in viewitems(phoc_histogram):
    print(k, v)
