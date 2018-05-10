#!/usr/bin/env python

from argparse import FileType

import numpy as np
import torch
from scipy.spatial.distance import cdist, pdist
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

import laia
import laia.logging as log
from dortmund_utils import build_dortmund_model
from laia.data import TextImageFromTextTableDataset
from laia.plugins.arguments import add_argument, add_defaults, args
from laia.utils import ImageToTensor

if __name__ == '__main__':
    add_defaults('gpu')
    add_argument('--phoc_levels', type=int, default=[1, 2, 3, 4, 5], nargs='+',
                 help='PHOC levels used to encode the transcript')
    add_argument('--distractors',
                 help='Transcription of each distractor image')
    add_argument('syms', help='Symbols table mapping from strings to integers')
    add_argument('img_dir', help='Directory containing word images')
    add_argument('queries', help='Transcription of each query image')
    add_argument('model_checkpoint', help='Filepath of the model checkpoint')
    add_argument('output', type=FileType('w'), help='Output file')
    args = args()

    syms = laia.utils.SymbolsTable(args.syms)
    phoc_size = sum(args.phoc_levels) * len(syms)
    model = build_dortmund_model(phoc_size)
    log.info('Model has {} parameters',
             sum(param.data.numel() for param in model.parameters()))
    model.load_state_dict(torch.load(args.model_checkpoint))
    model = model.cuda(args.gpu - 1) if args.gpu > 0 else model.cpu()
    model.eval()


    def process_image(sample):
        sample = Variable(sample, requires_grad=False)
        sample = sample.cuda(args.gpu - 1) if args.gpu > 0 else sample.cpu()
        phoc = torch.nn.functional.sigmoid(model(sample))
        return phoc.data.cpu().numpy()


    def process_dataset(filename):
        dataset = TextImageFromTextTableDataset(
            filename, args.img_dir, img_transform=ImageToTensor())
        data_loader = DataLoader(dataset)
        phocs = []
        samples = []
        for sample in tqdm(data_loader):
            phocs.append(process_image(sample['img']))
            samples.append(sample['id'][0])
        return np.concatenate(phocs), samples


    # Process queries
    query_phocs, query_ids = process_dataset(args.queries)

    Q = len(query_phocs)

    log.info('Computing pairwise distances among {} queries', Q)
    query_distances = pdist(query_phocs, 'braycurtis')
    # Sort pairs of examples in increasing order
    query_inds = [(i, j) for i in range(Q) for j in range(i + 1, Q)]
    query_inds = [(query_inds[k], k) for k in np.argsort(query_distances)]

    # Process distractors
    if args.distractors:
        distract_phocs, distract_ids = process_dataset(args.distractors)
        D = len(distract_phocs)
        log.info('Computing distances between {} queries and {} distractors',
                 Q, D)
        distract_distances = cdist(query_phocs, distract_phocs, 'braycurtis')
        distract_distances = distract_distances.reshape((Q * D,))
        distract_inds = [(i, j) for i in range(Q) for j in range(D)]
        distract_inds = [(distract_inds[k], k)
                         for k in np.argsort(distract_distances)]
    else:
        distract_distances = []
        distract_inds = []


    def print_query_dist(i, j, k):
        args.output.write('{} {} {}\n'.format(query_ids[i],
                                              query_ids[j],
                                              query_distances[k]))
        args.output.write('{} {} {}\n'.format(query_ids[j],
                                              query_ids[i],
                                              query_distances[k]))


    def print_distract_dist(i, j, k):
        args.output.write('{} {} {}\n'.format(query_ids[i],
                                              distract_ids[j],
                                              distract_distances[k]))


    qk, dk = 0, 0
    while qk < len(query_inds) and dk < len(distract_inds):
        if (query_distances[query_inds[qk][1]] <
                distract_distances[distract_inds[dk][1]]):
            (i, j), k = query_inds[qk]
            print_query_dist(i, j, k)
            qk += 1
        else:
            (i, j), k = distract_inds[dk]
            print_distract_dist(i, j, k)
            dk += 1

    while qk < len(query_inds):
        (i, j), k = query_inds[qk]
        print_query_dist(i, j, k)
        qk += 1

    while dk < len(distract_inds):
        (i, j), k = distract_inds[dk]
        print_distract_dist(i, j, k)
        dk += 1
