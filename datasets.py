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


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, data, targets):
        super().__init__()
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[[index]], self.targets[index].item()


class Grayscale2RGB(torch.utils.data.Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        if isinstance(self.dataset, (SupervisedDataSet, TestDataset)):
            img, label = self.dataset[index]
            img = img.repeat((3, 1, 1))
            return img, label
        else:
            img = self.dataset[index]
            img = img.repeat((3, 1, 1))
            return img


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

    data = dataset.data
    targets = dataset.targets
    if set_name.upper() == 'EMNIST-LETTERS':
        targets = targets - 1

    if train:
        indices = list(range(len(dataset)))
        random.shuffle(indices)

        cut_off = int(len(dataset) * supervised_ratio)

        sup_set = SupervisedDataSet(data, targets, indices[:cut_off])
        unsup_set = UnsupervisedDataSet(data, indices[cut_off:])
        if not is_grayscale:
            sup_set, unsup_set = Grayscale2RGB(
                sup_set), Grayscale2RGB(unsup_set)
        return sup_set, unsup_set
    else:
        test_set = TestDataset(data, targets)
        if not is_grayscale:
            test_set = Grayscale2RGB(test_set)
        return test_set


if __name__ == '__main__':
    sup_set, unsup_set = get_dataset('EMNIST-LETTERS', True, 0.3, True)
    # sup_set, unsup_set = get_dataset('MNIST', True, 0.3, False)
    # print(sup_set[0][0].size())
    # print(sup_set[0][1])
    # print(unsup_set[0].size())
    print(sup_set.targets.unique())
    # print(sup_set.dataset.targets)

    # test_set = get_dataset('MNIST', False, 0.3, True)
    # test_set = get_dataset('MNIST', False, 0.3, False)
    # print(test_set[0][0].size())
