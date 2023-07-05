import os
import glob
import copy
import torch
import torchvision
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler


IMAGENET_DEFAULT_MEAN = [0.485, 0.456, 0.406]
IMAGENET_DEFAULT_STD = [0.229, 0.224, 0.225]
CIFAR_DEFAULT_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR_DEFAULT_STD = [0.2023, 0.1994, 0.2010]


class myCIFARDataset(Dataset):
    def __init__(self, tensor_data, tensor_label, transform=None):
        self.data = tensor_data  # [N, 32, 32, 3]
        self.label = tensor_label
        assert tensor_data.shape[0] == tensor_label.shape[0]
        self.transform = transform

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        sample = self.data[idx]  # [32, 32, 3]
        sample = Image.fromarray(sample)
        label = self.label[idx]
        if self.transform:
            sample = self.transform(sample)  # what is the shape here: [3, 32, 32], somehow it flip the channel
        return sample, label


class CIFAR10:
    def __init__(self, args):
        super(CIFAR10, self).__init__()

        # use args:
        use_cuda = args.cuda
        num_workers = args.num_workers
        path = args.data_path
        batch_size = args.batch_size

        # basic information
        self.num_classes = 10
        self.input_dim = 32
        mean = CIFAR_DEFAULT_MEAN
        std = CIFAR_DEFAULT_STD
        self.input_channel = 3

        # Data loading code
        kwargs = {"num_workers": num_workers, "pin_memory": True} if use_cuda else {}
        normalize = transforms.Normalize(mean=mean, std=std)
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        dataset = datasets.CIFAR10(path, train=True, download=True, transform=transform)
        train_set, val_set = torch.utils.data.random_split(dataset, [45000, 5000], generator=torch.Generator().manual_seed(42))
        test_set = datasets.CIFAR10(
            path, train=False, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                normalize]),
        )

        self.train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=batch_size, shuffle=True, drop_last=False, **kwargs)
        self.val_loader = torch.utils.data.DataLoader(
            val_set, batch_size=batch_size, shuffle=False, drop_last=False, **kwargs)
        self.test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=batch_size, shuffle=False, drop_last=False, **kwargs)

