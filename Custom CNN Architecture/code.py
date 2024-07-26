import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiInputCNN(nn.Module):
    def __init__(self):
        super(MultiInputCNN, self).__init__()

        # Define the three CNN branches
        self.branch1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # Define the fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(256 * 3, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )

    def forward(self, x1, x2, x3):
        # Forward pass through each branch
        x1 = self.branch1(x1).view(x1.size(0), -1)
        x2 = self.branch2(x2).view(x2.size(0), -1)
        x3 = self.branch3(x3).view(x3.size(0), -1)

        # Concatenate the outputs from each branch
        x = torch.cat((x1, x2, x3), dim=1)

        # Forward pass through the fully connected layers
        x = self.fc(x)

        return x

# Example usage
model = MultiInputCNN()
input1 = torch.randn(8, 1, 64, 64)  # Batch size of 8, grayscale images of size 64x64
input2 = torch.randn(8, 1, 64, 64)
input3 = torch.randn(8, 1, 64, 64)
output = model(input1, input2, input3)
print(output.shape)  # Should be [8, 1]
