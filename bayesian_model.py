import torch
import torch.nn as nn
import torchbnn as bnn

class BayesianCNN(nn.Module):
    def __init__(self):
        super(BayesianCNN, self).__init__()
        self.conv1 = bnn.BayesConv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1, prior_mu=0.0, prior_sigma=0.1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = bnn.BayesConv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, prior_mu=0.0, prior_sigma=0.1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)

        self.flatten = nn.Flatten()
        self.fc1 = bnn.BayesLinear(in_features=64 * 37 * 37, out_features=128, prior_mu=0.0, prior_sigma=0.1)
        self.relu3 = nn.ReLU()
        self.fc2 = bnn.BayesLinear(in_features=128, out_features=1, prior_mu=0.0, prior_sigma=0.1)
        self.out = nn.Sigmoid()

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.out(self.fc2(x))
        return x
