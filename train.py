# -*- coding: utf-8 -*-

import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

from tqdm import tqdm

from dataset import get_loaders


class Siamese(nn.Module):

    def __init__(self):
        super(Siamese, self).__init__()
        self.c1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.c2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.c3 = nn.Conv2d(64, 128, kernel_size=3, padding=0)
        self.bn3 = nn.BatchNorm2d(128)

        self.fc4 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm2d(64)
        self.fc5 = nn.Linear(64, 2)

    def forward(self, x):
        h = F.max_pool2d(self.bn1(self.c1(x)), 2)
        h = F.max_pool2d(self.bn2(self.c2(h)), 2)
        h = F.avg_pool2d(self.bn3(self.c3(h)), 5)

        h = self.bn4(self.fc4(h.view(h.size(0), -1)))
        return self.fc5(h)


def contractive_loss(o1, o2, y):
    g, margin = F.pairwise_distance(o1, o2), 5.0
    loss = (1 - y) * (g ** 2) + y * (torch.clamp(margin - g, min=0) ** 2)
    return torch.mean(loss)


def main(args):
    # Set up dataset
    train_loader, test_loader = get_loaders(args.batch_size)

    model = Siamese().cuda()
    opt = optim.SGD(model.parameters(),
                    lr=args.lr,
                    momentum=0.9)
    scheduler = optim.lr_scheduler.MultiStepLR(opt, [5, 10], 0.1)
    cudnn.benckmark = True

    print("\t".join(["Epoch", "TrainLoss", "TestLoss"]))
    for e in range(args.epochs):
        scheduler.step()
        model.train()
        train_loss, train_n = 0, 0
        for x1, x2, y in tqdm(train_loader, total=len(train_loader), leave=False):
            x1, x2 = Variable(x1.cuda()), Variable(x2.cuda())
            y = Variable(y.float().cuda()).view(y.size(0), 1)

            o1, o2 = model(x1), model(x2)
            loss = contractive_loss(o1, o2, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss = loss.data[0] * y.size(0)
            train_n += y.size(0)

        model.eval()
        test_loss, test_n = 0, 0
        for x1, x2, y in tqdm(test_loader, total=len(test_loader), leave=False):
            x1, x2 = Variable(x1.cuda()), Variable(x2.cuda())
            y = Variable(y.float().cuda()).view(y.size(0), 1)

            o1, o2 = model(x1), model(x2)
            loss = contractive_loss(o1, o2, y)
            test_loss = loss.data[0] * y.size(0)
            test_n += y.size(0)
        if (e + 1) % 5 == 0:
            torch.save(model, "./checkpoint/{}.tar".format(e+1))
        print("{}\t{:.6f}\t{:.6f}".format(e, train_loss / train_n, test_loss / test_n))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.01)
    args = parser.parse_args()
    main(args)
