#!/usr/bin/env python

from argparse import FileType

import numpy as np
import torch
from scipy.spatial.distance import cdist, pdist
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

import laia
import laia.common.logging as log
from dortmund_utils import build_dortmund_model
from laia.data import TextImageFromTextTableDataset
from laia.common.arguments import add_argument, add_defaults, args
from laia.utils import ImageToTensor

if __name__ == '__main__':
    add_defaults('gpu')
    add_argument('--phoc_levels', type=int, default=[1, 2, 3, 4, 5], nargs='+',
                 help='PHOC levels used to encode the transcript')
    add_argument('syms', help='Symbols table mapping from strings to integers')
    add_argument('img_dir', help='Directory containing word images')
    add_argument('query_set', help='Transcription of each query image')
    add_argument('test_set', help='Transcription of each test image')
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
    query_phocs, query_ids = process_dataset(args.query_set)
    test_phocs, test_ids = process_dataset(args.test_set)

    distances = cdist(query_phocs, test_phocs, 'braycurtis')
    for i, qid in enumerate(query_ids):
        for j, tid in enumerate(test_ids):
            args.output.write('{} {} {}\n'.format(qid, tid, distances[i][j]))
