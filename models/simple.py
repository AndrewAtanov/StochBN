'''LeNet in PyTorch.'''
import torch.nn as nn
import torch.nn.functional as F
from stochbn import MyBatchNorm2d, MyBatchNorm1d


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.bn1 = MyBatchNorm2d(20)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.bn2 = MyBatchNorm2d(50)
        self.fc1 = nn.Linear(16*50, 500)
        self.bn3 = MyBatchNorm1d(500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(self.bn1(out))
        out = F.max_pool2d(out, 2)
        out = self.conv2(out)
        out = F.relu(self.bn2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), 16*50)
        out = self.fc1(out)
        out = F.relu(self.bn3(out))
        out = self.fc2(out)
        return out


class FC(nn.Module):
    def __init__(self):
        super(FC, self).__init__()
        self.fc1 = nn.Linear(28**2, 300)
        self.bn1 = MyBatchNorm1d(300)
        self.fc2 = nn.Linear(300, 100)
        self.bn2 = MyBatchNorm1d(100)
        self.classifier = nn.Linear(100, 10)

    def forward(self, x):
        out = x.view(x.size(0), 28**2)
        out = self.fc1(out)
        out = F.relu(self.bn1(out))
        out = self.fc2(out)
        out = F.relu(self.bn2(out))
        out = self.classifier(out)
        return out
