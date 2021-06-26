import torch
import torchvision


class MNIST(torch.utils.data.Dataset):
    def __init__(self, root, train, transform, download):
        super().__init__()
        self.mnist_dataset = torchvision.datasets.MNIST(
            root, train=train, transform=transform, download=download)
        self.train = train

    def __len__(self):
        if self.train:
            return self.mnist_dataset.train_data.size()[0]
        else:
            return self.mnist_dataset.test_data.size()[0]

    def __getitem__(self, index):
        if self.train:
            img, label = self.mnist_dataset.train_data[index], self.mnist_dataset.train_labels[index]
        else:
            img, label = self.mnist_dataset.test_data[index], self.mnist_dataset.test_labels[index]
        img = img.repeat((3, 1, 1))
        return img, label
