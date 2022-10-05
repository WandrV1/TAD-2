import sys
import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomNet(nn.Module):
    def __init__(self):
        super(CustomNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=12, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=12, out_channels=20, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=20, out_channels=24, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(in_features=24 * 14 * 14, out_features=2048)
        self.fc2 = nn.Linear(in_features=2048, out_features=512)
        self.fc3 = nn.Linear(in_features=512, out_features=128)
        self.out = nn.Linear(in_features=128, out_features=1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = x.reshape(-1, 24 * 14 * 14)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.out(x)
        return torch.sigmoid(x)


def get_model(name):
    if name == "resnet18":
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', weights='ResNet18_Weights.DEFAULT')
        for param in model.parameters():
            param.requires_grad = False
        model.fc = nn.Sequential(nn.Linear(512, 128),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(128, 1),
                                 nn.Sigmoid())
    elif name == "resnet50":
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', weights='ResNet50_Weights.DEFAULT')
        for param in model.parameters():
            param.requires_grad = False
        model.fc = nn.Sequential(nn.Linear(2048, 512),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(512, 128),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(128, 1),
                                 nn.Sigmoid())
    else:
        model = CustomNet()
    return model
