# evaluate.py

import torch
import torch.nn as nn
from models.vision_cnn import VisionCNN
from data.cifar_loader import get_cifar10_loaders
from sklearn.metrics import classification_report, accuracy_score, f1_score
import tempfile

def evaluate_and_log(version, change_desc, model_path="outputs/cifar10_cnn_v2.pth"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _, test_loader = get_cifar10_loaders(batch_size=64)

    model = VisionCNN(num_classes=10).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            all_preds.extend(outputs.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')

    print(f"Version: {version} | Change: {change_desc}")
    print(f"Accuracy: {acc:.4f}, Macro F1: {macro_f1:.4f}")

    # Append to experiments.md
    line = f"| {version} | {change_desc} | {acc:.3f} | {macro_f1:.3f} |  |\n"
    with open("experiments.md", "a") as f:
        f.write(line)

if __name__ == "__main__":
    # Adjust these parameters per run
    evaluate_and_log(version="v2.0", change_desc="BatchNorm after conv layers", 
                     model_path="outputs/cifar10_cnn_v2.pth")
