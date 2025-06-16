# utils/training.py

import torch
from torch import nn
from tqdm import tqdm

def train(model, train_loader, optimizer, loss_fn, device):
    model.train()
    total_loss, correct = 0, 0
    total = 0

    loop = tqdm(train_loader, desc="Training", leave=False)
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)

        # Forward
        outputs = model(images)
        loss = loss_fn(outputs, labels)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Stats
        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        # Update progress bar
        loop.set_postfix(loss=loss.item(), acc=100*correct/total)

    return total_loss / len(train_loader), correct / total
