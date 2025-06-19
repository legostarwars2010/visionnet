# VisionNet: CIFAR-10 Image Classifier from Scratch

🚀 Built entirely with PyTorch
🏗️ Deep CNN + Residual Blocks
📈 Achieves **81% test accuracy** on CIFAR-10 (v4.0)

---

## 🔍 Architecture Overview

A fully custom convolutional neural network designed for image classification:

* 3 convolutional blocks (32 → 64 → 128 filters)
* 2 residual blocks (ResNet-style)
* BatchNorm, ReLU, MaxPooling in each block
* Global Average Pooling + Fully Connected head
* Data augmentation: RandomCrop, HorizontalFlip
* Training on CPU (GPU-ready)

---

## 📊 Experiment Results

| Version | Accuracy | Notes                                     |
| ------- | -------- | ----------------------------------------- |
| v2.1    | 0.79     | Baseline CNN + BatchNorm                  |
| v3.0    | 0.79     | Added data augmentation                   |
| v3.1    | 0.79     | Added LR scheduler                        |
| v4.0    | **0.81** | Added residual blocks, global avg pooling |

---

## 🧪 Run It Yourself

### Install dependencies

```bash
pip install -r requirements.txt
```

### Train the model

```bash
python train.py
```

### Evaluate a saved model

```bash
python evaluate.py --model outputs/cifar10_cnn_v4.0.pth
```

---

## 📁 Project Structure

```
visionnet/
├── models/                # VisionCNN architecture variants
│   └── vision_cnn_v4.py
├── data/                  # CIFAR-10 loading + transforms
│   └── cifar_loader.py
├── utils/                 # Training and evaluation helpers
│   ├── training.py
│   └── evaluation.py
├── outputs/               # Saved models and plots
├── experiments.md         # Logged metrics per version
├── train.py               # Training script
├── evaluate.py            # Evaluation + classification report
├── requirements.txt       # Dependencies
└── README.md              # Project overview (this file)
```

---

## 📦 Features

* Clean modular codebase
* Easy to swap model versions
* Auto-logging to `experiments.md`
* Classification report after evaluation

---

## 🧠 Future Work

* Add SE attention blocks (v4.1?)
* Add transfer learning baseline
* Enable mixed-precision training with AMP (once GPU-ready)

---

## 🏁 License

MIT License

