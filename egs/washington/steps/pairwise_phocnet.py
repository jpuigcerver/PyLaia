import argparse
from collections import Counter

import laia
import laia.logging as log
import numpy as np
import torch
from dortmund_utils import build_dortmund_model
from laia.data import TextImageFromTextTableDataset
from laia.hooks.meters.pairwise_average_precision_meter import \
    PairwiseAveragePrecisionMeter
from laia.plugins.arguments import add_argument, add_defaults, args
from laia.utils import ImageToTensor
from scipy.spatial.distance import pdist
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

if __name__ == '__main__':
    add_defaults('gpu')
    add_argument('--phoc_levels', type=int, default=[1, 2, 3, 4, 5], nargs='+',
                 help='PHOC levels used to encode the transcript')
    add_argument('syms', help='Symbols table mapping from strings to integers')
    add_argument('img_dir', help='Directory containing word images')
    add_argument('queries', help='Transcription of each query image')
    add_argument('model_checkpoint', help='Filepath of the model checkpoint')
    add_argument('output', type=argparse.FileType('w'),
                 help='Filepath of the output file')
    args = args()

    syms = laia.utils.SymbolsTable(args.syms)
    phoc_size = sum(args.phoc_levels) * len(syms)
    model = build_dortmund_model(phoc_size)
    log.info('Model has {} parameters',
             sum(param.data.numel() for param in model.parameters()))
    model.load_state_dict(torch.load(args.model_checkpoint))
    model = model.cuda(args.gpu - 1) if args.gpu > 0 else model.cpu()
    model.eval()

    queries_dataset = TextImageFromTextTableDataset(
        args.queries, args.img_dir, img_transform=ImageToTensor())
    queries_loader = DataLoader(queries_dataset)

    meter = PairwiseAveragePrecisionMeter(metric='braycurtis')


    def process_image(sample):
        sample = Variable(sample, requires_grad=False)
        sample = sample.cuda(args.gpu - 1) if args.gpu > 0 else sample.cpu()
        phoc = torch.nn.functional.sigmoid(model(sample))
        return phoc.data.cpu().numpy()


    # Predict PHOC vectors
    phocs = []
    labels = []
    samples = []
    for query in tqdm(queries_loader):
        phocs.append(process_image(query['img']))
        labels.append(query['txt'][0])
        samples.append(query['id'][0])
        meter.add(phocs[-1], query['txt'][0])

    log.info('Internal gAP: {meter.value[0]:5.1%}, mAP: {meter.value[1]:5.1%}',
             meter=meter)

    # Filter to use only labels with at least two samples
    valid_phocs = []
    valid_labels = []
    valid_samples = []
    label_count = Counter(labels)
    for sample_id, label, phoc in zip(samples, labels, phocs):
        if label_count[label] > 1:
            valid_phocs.append(phoc)
            valid_labels.append(label)
            valid_samples.append(sample_id)

    n = len(valid_phocs)
    log.info('Computing pairwise distances among {} queries', n)
    distances = pdist(np.concatenate(valid_phocs), 'braycurtis')
    # Sort pairs of examples in increasing order
    inds = [(i, j) for i in range(n) for j in range(i + 1, n)]
    inds = [(inds[k], k) for k in np.argsort(distances)]

    for (i, j), k in inds:
        args.output.write('{} {} {}\n'.format(valid_samples[i],
                                              valid_samples[j],
                                              distances[k]))
        args.output.write('{} {} {}\n'.format(valid_samples[j],
                                              valid_samples[i],
                                              distances[k]))
