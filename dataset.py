# -*- coding: utf-8 -*-

import random

import torch

from torchvision import datasets
from torchvision import transforms

from PIL import Image


class FashionMNIST(datasets.FashionMNIST):

    def __init__(self, *args, **kwargs):
        super(FashionMNIST, self).__init__(*args, **kwargs)
        if kwargs["train"] is True:
            self.data, self.labels = self.train_data, self.train_labels
        else:
            self.data, self.labels = self.test_data, self.test_labels

    def __getitem__(self, idx):
        x1, t1 = self.data[idx], self.labels[idx]

        is_diff = random.randint(0, 1)
        while True:
            idx2 = random.randint(0, len(self)-1)
            x2, t2 = self.data[idx2], self.labels[idx2]
            if is_diff and t1 != t2:
                break
            if not is_diff and t1 == t2:
                break

        x1, x2 = Image.fromarray(x1.numpy()), Image.fromarray(x2.numpy())
        if self.transform is not None:
            x1, x2 = self.transform(x1), self.transform(x2)
        return x1, x2, int(is_diff)


def get_loaders(batch_size):
    train_loader = torch.utils.data.DataLoader(
        FashionMNIST("./data", train=True, download=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()])),
        batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        FashionMNIST("./data", train=False, download=False, transform=transforms.Compose([
            transforms.ToTensor()])),
        batch_size=batch_size, shuffle=False, pin_memory=True)
    return train_loader, test_loader
