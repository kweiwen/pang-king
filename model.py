import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.nn.functional as F


class CustomModel(nn.Module):
    def __init__(self, spatial_feature=13, material_features=36, hidden_dim=256):
        super(CustomModel, self).__init__()
        self.max_size = 32768

        self.fc1 = nn.Linear(spatial_feature, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)

        # Regression Part
        self.fc5 = nn.Linear(spatial_feature + 1 + material_features, 256)
        self.bn5 = nn.BatchNorm1d(256)

        self.fc6 = nn.Linear(256, 512)
        self.bn6 = nn.BatchNorm1d(512)

        self.fc7 = nn.Linear(512, self.max_size)

    def forward(self, dimension, tx, rx, order, material):

        dist = torch.sqrt(torch.sum((tx - rx) ** 2, dim=1))
        n = order.int()
        images = 1 + (2 * n * (2 * n * n + 3 * n + 4) / 3)

        x0 = torch.cat((dimension, tx, rx, dist.unsqueeze(1), order.unsqueeze(1), images.unsqueeze(1), dimension.prod(dim=1).unsqueeze(1)), dim=1)
        x1 = F.relu(self.fc1(x0))
        x2 = F.relu(self.fc2(x1))
        x3 = F.relu(self.fc3(x2))
        x4 = F.relu(self.fc4(x3))
        size_out = x4.squeeze()

        # Concatenate size_out, x, and material for rir_seq prediction
        # combined = torch.cat(
        #     (dimension, tx, rx, order.unsqueeze(1), dist.unsqueeze(1), images.unsqueeze(1), x3, material),
        #     dim=1)
        #
        # y1 = torch.sigmoid(self.fc5(combined))
        # y1 = self.bn5(y1)
        #
        # y2 = torch.sigmoid(self.fc6(y1))
        # y2 = self.bn6(y2)
        #
        # rir_seq = self.fc7(y2)
        rir_seq = []

        return size_out, rir_seq

