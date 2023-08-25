import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomModel(nn.Module):
    def __init__(self, input_features=10, material_features=36, hidden_dim=256, batch_size=16):
        super(CustomModel, self).__init__()
        self.max_size = 32768

        # Transformer Part
        # encoder_layers = nn.TransformerEncoderLayer(d_model=input_features, nhead=1)
        # self.transformer = nn.TransformerEncoder(encoder_layers, num_layers=1)
        self.fc1 = nn.Linear(input_features, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)

        # Regression Part
        self.fc5 = nn.Linear(64 + material_features, hidden_dim)
        self.fc6 = nn.Linear(hidden_dim, self.max_size)

    def forward(self, x, y):
        # x0 = torch.sigmoid(self.transformer(x.unsqueeze(1)))
        x1 = torch.relu(self.fc1(x))
        x2 = torch.relu(self.fc2(x1))
        x3 = torch.relu(self.fc3(x2))
        x4 = torch.relu(self.fc4(x3))
        size_out = x4.squeeze()

        noise = torch.rand(x.size(0), self.max_size) * 2 - 1  # Changing noise range to [-1,1]

        feature = torch.cat((x2.squeeze(1), y), dim=1)

        # Use the sigmoid function to limit the mask values between [0,1]
        mask = torch.tanh(self.fc6(torch.relu(self.fc5(feature))))

        # Multiply the mask with noise
        masked_noise = mask * noise

        return size_out, masked_noise

