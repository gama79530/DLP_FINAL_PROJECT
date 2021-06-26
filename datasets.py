import torch
import torchvision
import os
import random


class SupervisedDataSet(torch.utils.data.Dataset):
    def __init__(self, data, targets, indices):
        super().__init__()
        self.data = data
        self.targets = targets
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.data[[self.indices[index]]], self.targets[self.indices[index]].item()


class UnsupervisedDataSet(torch.utils.data.Dataset):
    def __init__(self, data, indices):
        super().__init__()
        self.data = data
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.data[[self.indices[index]]]


class Grayscale2RGB(torch.utils.data.Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        if isinstance(self.dataset, SupervisedDataSet):
            img, label = self.dataset[index]
            img = img.repeat((3, 1, 1))
            return img, label

        else:
            img = self.dataset[index]
            img = img.repeat((3, 1, 1))
            return img, label


def get_dataset(set_name, train, supervised_ratio=0.2, is_grayscale=True, fix_seed=True):
    """
        set_name: MNIST, EMNIST-LETTERS, FASHION-MNIST
    """
    if fix_seed:
        # for reproducibility
        random.seed(777)

    root = os.path.join('', 'datasets')
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5), (0.5))
    ])
    download = True
    if set_name.upper() == 'MNIST':
        dataset = torchvision.datasets.MNIST(
            root, train=train, transform=transform, download=download)
    elif set_name.upper() == 'EMNIST-LETTERS':
        dataset = torchvision.datasets.EMNIST(
            root, split='letters', train=train, transform=transform, download=download)
    elif set_name.upper() == 'FASHION-MNIST':
        dataset = torchvision.datasets.FashionMNIST(
            root, train=train, transform=transform, download=download)

    if not train:
        return dataset

    indices = list(range(len(dataset)))
    random.shuffle(indices)

    data = dataset.data
    targets = dataset.targets

    cut_off = int(len(dataset) * supervised_ratio)

    if is_grayscale:
        return SupervisedDataSet(data, targets, indices[:cut_off]), UnsupervisedDataSet(data, indices[cut_off:])
    else:
        return Grayscale2RGB(SupervisedDataSet(data, targets, indices[:cut_off])), Grayscale2RGB(UnsupervisedDataSet(data, indices[cut_off:]))
