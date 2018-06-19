from __future__ import print_function

import argparse

import torch
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
    add_argument("gt_txt", help="Transcription of each image")
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

    dataset = TextImageFromTextTableDataset(
        args.gt_txt, args.img_dir, img_transform=ImageToTensor()
    )
    loader = DataLoader(dataset)

    def process_image(sample):
        sample = Variable(sample, requires_grad=False)
        sample = sample.cuda(args.gpu - 1) if args.gpu > 0 else sample.cpu()
        phoc = torch.nn.functional.sigmoid(model(sample))
        return phoc.data.cpu().numpy()

    # Predict PHOC vectors
    for query in tqdm(loader):
        phoc = process_image(query["img"])
        print(query["id"][0], file=args.output, end="")
        for j in range(phoc.shape[1]):
            print(" %.12g" % phoc[0, j], file=args.output, end="")
        print("", file=args.output)
