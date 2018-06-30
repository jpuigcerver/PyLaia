#!/usr/bin/env python

from __future__ import division

import argparse

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

import laia
import laia.common.logging as log
from laia.data import TextImageFromTextTableDataset
from laia.models.kws.dortmund_phocnet import DortmundPHOCNet
from laia.common.arguments import add_argument, add_defaults, args
from laia.utils import ImageToTensor
from laia.utils.phoc import cphoc, pphoc

if __name__ == "__main__":
    add_defaults("gpu")
    add_argument(
        "--phoc_levels",
        type=int,
        default=[1, 2, 3, 4, 5],
        nargs="+",
        help="PHOC levels used to encode the transcript",
    )
    add_argument(
        "--tpp_levels",
        type=int,
        default=[1, 2, 3, 4, 5],
        nargs="*",
        help="Temporal Pyramid Pooling levels",
    )
    add_argument(
        "--spp_levels",
        type=int,
        default=None,
        nargs="*",
        help="Spatial Pyramid Pooling levels",
    )
    add_argument("--distractors", help="Transcription of each distractor image")
    add_argument("syms", help="Symbols table mapping from strings to integers")
    add_argument("img_dir", help="Directory containing word images")
    add_argument("queries", help="Transcription of each query image")
    add_argument("model_checkpoint", help="Filepath of the model checkpoint")
    add_argument("output", type=argparse.FileType("w"), help="Output file")
    args = args()

    syms = laia.utils.SymbolsTable(args.syms)
    phoc_size = sum(args.phoc_levels) * len(syms)
    model = DortmundPHOCNet(
        phoc_size=phoc_size, tpp_levels=args.tpp_levels, spp_levels=args.spp_levels
    )
    log.info(
        "Model has {} parameters",
        sum(param.data.numel() for param in model.parameters()),
    )
    model.load_state_dict(torch.load(args.model_checkpoint))
    model = model.cuda(args.gpu - 1) if args.gpu > 0 else model.cpu()
    model.eval()

    def process_image(sample):
        sample = Variable(sample, requires_grad=False)
        sample = sample.cuda(args.gpu - 1) if args.gpu > 0 else sample.cpu()
        phoc = torch.nn.functional.logsigmoid(model(sample))
        return phoc.data.cpu().squeeze()

    def process_dataset(filename):
        dataset = TextImageFromTextTableDataset(
            filename, args.img_dir, img_transform=ImageToTensor()
        )
        data_loader = DataLoader(dataset)
        phocs = []
        samples = []
        for sample in tqdm(data_loader):
            phocs.append(process_image(sample["img"]))
            samples.append(sample["id"][0])
        return torch.stack(phocs).type("torch.DoubleTensor"), samples

    # Process queries
    phocs, samples = process_dataset(args.queries)
    n = len(phocs)
    log.info("Computing pairwise relevance probabilities among {} queries", n)
    logprobs = pphoc(phocs)
    for i in range(n):
        for j in range(i + 1, n):  # Note: this skips the pair (i, i)
            k = i * n - i * (i - 1) // 2 + (j - i)
            args.output.write("{} {} {}\n".format(samples[i], samples[j], logprobs[k]))
            args.output.write("{} {} {}\n".format(samples[j], samples[i], logprobs[k]))

    # Process distractors
    if args.distractors:
        phocs2, samples2 = process_dataset(args.distractors)
        n2 = len(phocs2)
        log.info("Computing distances between {} queries and {} distractors", n, n2)
        logprobs = cphoc(phocs, phocs2)
        for i in range(n):
            for j in range(n2):
                args.output.write(
                    "{} {} {}\n".format(samples[i], samples2[j], logprobs[i, j])
                )

    log.info("Done.")
