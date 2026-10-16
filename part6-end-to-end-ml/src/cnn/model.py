import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(41)


# A small LeNet-style CNN for 28x28 MNIST images: two conv+ReLU+maxpool
# blocks, then three linear layers and a log-softmax over ten classes.
# (Weights are random — the point is the compilation pipeline, not accuracy.)
class ConvolutionalNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 3, 1)
        self.conv2 = nn.Conv2d(6, 16, 3, 1)
        # Fully connected layers
        self.fc1 = nn.Linear(5 * 5 * 16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, 2, 2)  # 2x2 kernel, stride 2
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2, 2)
        # Flatten; -1 keeps the batch dimension free
        X = X.view(-1, 16 * 5 * 5)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)
        return F.log_softmax(X, dim=1)
