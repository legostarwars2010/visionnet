# evaluate.py

import torch
import torch.nn as nn
from models.vision_cnn import VisionCNN
from data.cifar_loader import get_cifar10_loaders
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def evaluate():
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load test data
    _, test_loader = get_cifar10_loaders(batch_size=64)

    # Load model
    model = VisionCNN(num_classes=10).to(device)
    model.load_state_dict(torch.load("outputs/cifar10_cnn.pth"))
    model.eval()

    all_preds = []
    all_labels = []

    # Inference loop
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Metrics
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=[
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]))

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[
                    'airplane', 'auto', 'bird', 'cat', 'deer',
                    'dog', 'frog', 'horse', 'ship', 'truck'
                ],
                yticklabels=[
                    'airplane', 'auto', 'bird', 'cat', 'deer',
                    'dog', 'frog', 'horse', 'ship', 'truck'
                ])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    evaluate()
