import argparse

import numpy as np
import torch
from scipy.spatial.distance import cdist
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

import laia
import laia.logging as log
from laia.data import TextImageFromTextTableDataset
from laia.models.kws.dortmund_phocnet import DortmundPHOCNet
from laia.plugins.arguments import add_argument, add_defaults, args
from laia.utils import ImageToTensor

if __name__ == "__main__":
    add_defaults("gpu")
    add_argument(
        "--phoc_levels",
        type=int,
        default=[1, 2, 3, 4, 5],
        nargs="+",
        help="PHOC levels used to encode the transcript",
    )
    add_argument("syms", help="Symbols table mapping from strings to integers")
    add_argument("img_dir", help="Directory containing word images")
    add_argument("candidates", help="Transcription of each candidate image")
    add_argument("queries", help="Transcription of each query image")
    add_argument("model_checkpoint", help="Filepath of the model checkpoint")
    add_argument(
        "output", type=argparse.FileType("w"), help="Filepath of the output file"
    )
    args = args()

    syms = laia.utils.SymbolsTable(args.syms)
    phoc_size = sum(args.phoc_levels) * len(syms)
    model = DortmundPHOCNet(phoc_size)
    log.info(
        "Model has {} parameters",
        sum(param.data.numel() for param in model.parameters()),
    )
    model.load_state_dict(torch.load(args.model_checkpoint))
    model = model.cuda(args.gpu - 1) if args.gpu > 0 else model.cpu()
    model.eval()

    candidates_dataset = TextImageFromTextTableDataset(
        args.candidates, args.img_dir, img_transform=ImageToTensor()
    )
    candidates_loader = DataLoader(candidates_dataset)

    queries_dataset = TextImageFromTextTableDataset(
        args.queries, args.img_dir, img_transform=ImageToTensor()
    )
    queries_loader = DataLoader(queries_dataset)

    def process_image(sample):
        sample = Variable(sample, requires_grad=False)
        sample = sample.cuda(args.gpu - 1) if args.gpu > 0 else sample.cpu()
        phoc = torch.nn.functional.sigmoid(model(sample))
        return phoc.data.cpu().numpy()

    ## Predict PHOC vectors
    # (a) Candidates
    candidate_phocs = []
    candidate_labels = []
    candidate_samples = []
    for candidate in tqdm(candidates_loader):
        candidate_phocs.append(process_image(candidate["img"]))
        candidate_labels.append(candidate["txt"][0])
        candidate_samples.append(candidate["id"][0])
    # (b) Queries
    query_phocs = []
    query_labels = []
    query_samples = []
    for query in tqdm(queries_loader):
        query_phocs.append(process_image(query["img"]))
        query_labels.append(query["txt"][0])
        query_samples.append(query["id"][0])

    log.info(
        "Computing pairwise distances among {} queries and {} candidates",
        len(query_samples),
        len(candidate_samples),
    )
    distances = cdist(
        np.concatenate(query_phocs), np.concatenate(candidate_phocs), "braycurtis"
    )

    for i in range(len(query_samples)):
        for j in range(len(candidate_samples)):
            args.output.write(
                "{} {} {}\n".format(
                    query_samples[i], candidate_samples[j], distances[i, j]
                )
            )
