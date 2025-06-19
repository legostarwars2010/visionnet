# evaluate.py

import torch
import torch.nn as nn
from models.vision_cnn import VisionCNN
from data.cifar_loader import get_cifar10_loaders
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_and_log(version, change_desc, model_path="outputs/cifar10_cnn_v4.pth"):
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

    # Accuracy and F1
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(
        all_labels, all_preds,
        target_names=[
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]
    ))

    # Plot confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=[
                    'airplane', 'automobile', 'bird', 'cat', 'deer',
                    'dog', 'frog', 'horse', 'ship', 'truck'
                ],
                yticklabels=[
                    'airplane', 'automobile', 'bird', 'cat', 'deer',
                    'dog', 'frog', 'horse', 'ship', 'truck'
                ])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

    # Markdown log line
    log_line = f"| {version} | {change_desc} | {acc:.3f} | {macro_f1:.3f} |  |\n"
    with open("experiments.md", "a") as f:
        f.write(log_line)

    print(f"\n✅ Results logged to experiments.md (version {version})")

if __name__ == "__main__":
    # Change this per version
    evaluate_and_log(
    version="v4",
    change_desc="Added residual blocks to conv layers",
    model_path="outputs/cifar10_cnn_v4.pth"
)
