# models/vision_cnn.py

import torch.nn as nn

class VisionCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(VisionCNN, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # 3 input channels (RGB)\
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # downsample to 16x16

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # downsample to 8x8
        )

        self.fc_block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = self.fc_block(x)
        return x
