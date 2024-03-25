import torch
import torchvision
from torchvision.transforms import InterpolationMode
import torchvision.transforms as transforms
from torch.utils.data import Dataset

import numpy as np

import matplotlib.pyplot as plt


class CIFARDataset(Dataset):
    def __init__(
        self,
        train_set: bool = True,
        test_set: bool = False,
        test_sample_size: int = None,
    ):
        super().__init__()
        self.train_set = train_set
        self.data = None
        self.test_data = None
        self.classes = (
            "plane",
            "car",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        )
        if train_set:
            self.data = torchvision.datasets.CIFAR10(
                root="../data", train=self.train_set, download=True
            )

            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize((32, 32)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomResizedCrop(
                        (32, 32),
                        scale=(0.8, 1.0),
                        ratio=(0.75, 1.3333333333333333),
                        interpolation=InterpolationMode.BILINEAR,
                    ),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
        else:
            self.data = torchvision.datasets.CIFAR10(
                root="../data", train=False, download=True
            )
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize((32, 32)),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )

            if test_set:
                self.test_indices = torch.randperm(len(self.data))[:test_sample_size]
                self.test_set = torch.utils.data.Subset(self.data, self.test_indices)

    def _show_images(self):
        # Pick 30 samples randomly
        indices = torch.randperm(len(self.data))[:30]
        images = [np.asarray(self.data[i][0]) for i in indices]
        labels = [self.data[i][1] for i in indices]
        # Visualize the images using matplotlib
        fig = plt.figure(figsize=(10, 10))
        for i in range(30):
            ax = fig.add_subplot(6, 5, i + 1, xticks=[], yticks=[])
            ax.imshow(images[i])
            ax.set_title(self.classes[labels[i]])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.transform(self.data[item])
