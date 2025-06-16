from data.cifar_loader import get_cifar10_loaders

train_loader, test_loader = get_cifar10_loaders()

for images, labels in train_loader:
    print("Image batch shape:", images.shape)
    print("Label batch shape:", labels.shape)
    break
