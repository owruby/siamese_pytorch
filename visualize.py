# -*- coding: utf-8 -*-

import argparse

import torch
from torch.autograd import Variable

from torchvision import datasets
from torchvision import transforms

import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import offsetbox
import seaborn as sns

from train import Siamese


def main(args):
    sns.set(style="whitegrid", font_scale=1.5)

    test_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST("./data", train=False, download=False, transform=transforms.Compose([
            transforms.ToTensor()
        ])), batch_size=100, shuffle=True)
    model = torch.load(args.ck_path)
    model.eval()
    inputs, embs, targets = [], [], []
    for x, t in tqdm(test_loader, total=len(test_loader)):
        x = Variable(x.cuda())
        o1 = model(x)
        inputs.append(x.cpu().data.numpy())
        embs.append(o1.cpu().data.numpy())
        targets.append(t.numpy())

    inputs = np.array(inputs).reshape(-1, 28, 28)
    embs = np.array(embs).reshape((-1, 2))
    targets = np.array(targets).reshape((-1,))

    n_plots = args.n_plots
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111)
    ax.set_title("FashionMNIST 2D embeddigs")
    for x, e, t in zip(inputs[:n_plots], embs[:n_plots], targets[:n_plots]):
        imagebox = offsetbox.AnnotationBbox(
            offsetbox.OffsetImage(x, zoom=0.5, cmap=plt.cm.gray_r),
            xy=e, frameon=False)
        ax.add_artist(imagebox)
    
    ax.set_xlim(embs[:, 0].min(), embs[:, 0].max())
    ax.set_ylim(embs[:, 1].min(), embs[:, 1].max())
    plt.tight_layout()
    plt.savefig("vis.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ck_path", type=str, default="./checkpoint/20.tar")
    parser.add_argument("--n_plots", type=int, default=500)
    args = parser.parse_args()
    main(args)
