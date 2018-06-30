import argparse

import numpy as np
import torch
from scipy.spatial.distance import pdist
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
    add_argument("syms", help="Symbols table mapping from strings to integers")
    add_argument("img_dir", help="Directory containing word images")
    add_argument("queries", help="Transcription of each query image")
    add_argument("model_checkpoint", help="Filepath of the model checkpoint")
    add_argument(
        "output", type=argparse.FileType("w"), help="Filepath of the output file"
    )
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

    queries_dataset = TextImageFromTextTableDataset(
        args.queries, args.img_dir, img_transform=ImageToTensor()
    )
    queries_loader = DataLoader(queries_dataset)

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
        phocs.append(process_image(query["img"]))
        labels.append(query["txt"][0])
        samples.append(query["id"][0])

    n = len(phocs)
    log.info("Computing pairwise distances among {} queries", n)
    distances = pdist(np.concatenate(phocs), "braycurtis")
    # Sort pairs of examples in increasing order
    inds = [(i, j) for i in range(n) for j in range(i + 1, n)]
    inds = [(inds[k], k) for k in np.argsort(distances)]

    for (i, j), k in inds:
        args.output.write("{} {} {}\n".format(samples[i], samples[j], distances[k]))
        args.output.write("{} {} {}\n".format(samples[j], samples[i], distances[k]))
    log.info("Done.")
