# train.py

import torch
import torch.nn as nn
import torch.optim as optim

from models.vision_cnn import VisionCNN
from data.cifar_loader import get_cifar10_loaders
from utils.training import train

def main():
    # ✅ Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ✅ Data
    train_loader, test_loader = get_cifar10_loaders(batch_size=64)

    # ✅ Model
    model = VisionCNN(num_classes=10).to(device)

    # ✅ Loss and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    from torch.optim.lr_scheduler import StepLR
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)


    # ✅ Training loop
    epochs = 10
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        train_loss, train_acc = train(model, train_loader, optimizer, loss_fn, device)
        scheduler.step()
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"Train Loss: {train_loss:.4f}, Accuracy: {train_acc*100:.2f}%")

    # ✅ Save the model
    torch.save(model.state_dict(), "outputs/cifar10_cnn_v4.pth")
    print("\nModel saved to outputs/cifar10_cnn_v4.pth")

if __name__ == "__main__":
    main()
