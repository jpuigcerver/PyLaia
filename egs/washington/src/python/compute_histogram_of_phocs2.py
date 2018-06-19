from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

from future.utils import viewitems

from laia.utils.phoc import unigram_phoc

parser = argparse.ArgumentParser()
parser.add_argument('input_text', type=argparse.FileType('r'))
args = parser.parse_args()

# Load George Washington vocabulary
vocabulary = set()
for line in args.input_text:
    word = tuple(line.split()[1:])
    vocabulary.add(word)

# Recover George Washington alphabet
alphabet = set()
for word in vocabulary:
    alphabet.update([ch for ch in word])
alphabet = sorted(list(alphabet))

# Obtain the different PHOCs and count how many words produce the
# same PHOC code.
unigram_map = {c: i for i, c in enumerate(alphabet)}

phoc_levels = 1
done = False
while not done:
    phoc_counter = {}
    for word in vocabulary:
        phoc = unigram_phoc(word, unigram_map, list(range(1, phoc_levels + 1)))
        if phoc in phoc_counter:
            phoc_counter[phoc] += 1
        else:
            phoc_counter[phoc] = 1

    # Compute PHOC histogram:
    unique_phocs = [phoc for phoc, counter in viewitems(phoc_counter)
                    if counter == 1]
    print(phoc_levels, len(unique_phocs), len(unique_phocs) / len(vocabulary))
    phoc_levels += 1
    if len(unique_phocs) == len(vocabulary):
        done = True
