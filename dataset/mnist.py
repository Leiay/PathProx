# import os
# import copy
import torch
# import numpy as np
# from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import Dataset


class MNIST:
    def __init__(self, args):
        """
        use args: num_workers, cuda, data_path, batch_size
        """
        super(MNIST, self).__init__()

        # use args:
        use_cuda = args.cuda
        num_workers = args.num_workers
        path = args.data_path
        batch_size = args.batch_size

        # basic information
        self.input_dim = 28
        self.num_classes = 10
        self.input_channel = 1

        # Data loading code
        kwargs = {"num_workers": num_workers, "pin_memory": True} if use_cuda else {}
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset = datasets.MNIST(path, train=True, download=True, transform=transform)
        train_set, val_set = torch.utils.data.random_split(dataset, [55000, 5000], generator=torch.Generator().manual_seed(42))
        test_set = datasets.MNIST(path, train=False, transform=transform)
        self.train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=batch_size, shuffle=True, drop_last=False, **kwargs)
        self.val_loader = torch.utils.data.DataLoader(
            val_set, batch_size=batch_size, shuffle=False, drop_last=False, **kwargs)
        self.test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=batch_size, shuffle=False, drop_last=False, **kwargs)


class MNIST_subset:
    def __init__(self, args):
        """
        use args: num_workers, cuda, data_path, batch_size
        """
        super(MNIST_subset, self).__init__()

        # use args:
        use_cuda = args.cuda
        num_workers = args.num_workers
        path = args.data_path
        batch_size = args.batch_size

        # basic information
        self.input_dim = 28
        self.num_classes = 10
        self.input_channel = 1
        self.per_class_num = 100
        label_include = 10
        # in total per_class_num * label_include number of samples

        # Data loading code
        kwargs = {"num_workers": num_workers, "pin_memory": True} if use_cuda else {}
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_set = datasets.MNIST(path, train=True, download=True, transform=transform)
        val_set = datasets.MNIST(path, train=True, download=True, transform=transform)

        # permute
        train_data = train_set.__dict__['data']
        train_targets = train_set.__dict__['targets']
        idx_rand = torch.randperm(60000, generator=torch.Generator().manual_seed(42))
        train_data = train_data[idx_rand]
        train_targets = train_targets[idx_rand]
        train_set.__dict__['data'] = train_data
        train_set.__dict__['targets'] = train_targets

        updated_train_data, updated_train_targets = [], []
        updated_val_data, updated_val_targets = [], []  # val set are those samples that are not included in the train
        count = torch.zeros(label_include)
        for i in range(60000):
            target_i = train_set.__dict__['targets'][i].item()
            sample_i = train_set.__dict__['data'][i]
            if target_i in torch.arange(label_include).tolist() and count[target_i] < self.per_class_num:
                updated_train_data.append(sample_i)
                updated_train_targets.append(target_i)
                count[target_i] += 1
            else:
                updated_val_data.append(sample_i)
                updated_val_targets.append(target_i)

        train_set.__dict__['targets'] = updated_train_targets
        train_set.__dict__['data'] = updated_train_data
        # train_set = corrupt_this_dataset(train_set, label_corruption) if label_corruption > 0 else train_set
        val_set.__dict__['targets'] = updated_val_targets
        val_set.__dict__['data'] = updated_val_data
        # train_set = my_dataset(updated_train_data, updated_train_targets)
        # val_set = my_dataset(updated_val_data, updated_val_targets)

        test_set = datasets.MNIST(path, train=False, transform=transform)

        updated_test_data = []
        updated_test_targets = []
        for i in range(10000):
            target_i = test_set.__dict__['targets'][i].item()
            if target_i in torch.arange(label_include).tolist():
                sample_i = test_set.__dict__['data'][i]
                updated_test_data.append(sample_i)
                updated_test_targets.append(target_i)

        # test_set = my_dataset(updated_test_data, updated_test_targets)
        test_set.__dict__['targets'] = updated_test_targets
        test_set.__dict__['data'] = updated_test_data

        self.train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=batch_size, shuffle=True, drop_last=False, **kwargs)
        self.val_loader = torch.utils.data.DataLoader(
            val_set, batch_size=batch_size, shuffle=False, drop_last=False, **kwargs)
        self.test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=batch_size, shuffle=False, drop_last=False, **kwargs)
        self.lips_loader = torch.utils.data.DataLoader(
            test_set, batch_size=1, shuffle=False, drop_last=False, **kwargs)
