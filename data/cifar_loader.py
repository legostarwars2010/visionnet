# data/cifar_loader.py

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_cifar10_loaders(batch_size=64):
    # Define standard transforms
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),  # mean per channel
                             (0.5, 0.5, 0.5))  # std per channel
    ])

    # Download and load the datasets
    train_dataset = datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    # Wrap them in DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
