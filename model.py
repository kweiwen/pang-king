import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomModel(nn.Module):
    def __init__(self, input_features=10, material_features=36, hidden_dim=256):
        super(CustomModel, self).__init__()
        self.max_size = 32768

        # Transformer Part
        self.fc1 = nn.Linear(input_features, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)

        # Regression Part
        self.fc5 = nn.Linear(1 + input_features + material_features, hidden_dim)
        self.fc6 = nn.Linear(hidden_dim, hidden_dim)
        self.fc7 = nn.Linear(hidden_dim, self.max_size)

    def forward(self, x, material):
        x1 = torch.relu(self.fc1(x))
        x2 = torch.relu(self.fc2(x1))
        x3 = torch.relu(self.fc3(x2))
        x4 = torch.relu(self.fc4(x3))
        size_out = x4.squeeze()

        # Concatenate size_out, x, and material for rir_seq prediction
        combined = torch.cat((size_out.unsqueeze(1), x, material), dim=1)
        y1 = torch.relu(self.fc5(combined))
        y2 = torch.relu(self.fc6(y1))
        y3 = self.fc7(y2)

        # Mask the output using size_out
        batch_size = x.size(0)
        mask = torch.arange(self.max_size).expand(batch_size, self.max_size).to(x.device)
        mask = mask < size_out.unsqueeze(1)
        rir_seq = y3 * mask.float()

        return size_out, rir_seq

