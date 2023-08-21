import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomModel(nn.Module):
    def __init__(self, input_features=14, material_features=36, hidden_dim=256):
        super(CustomModel, self).__init__()

        # Transformer Part
        encoder_layers = nn.TransformerEncoderLayer(d_model=input_features, nhead=1)
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers=2)
        self.fc1 = nn.Linear(input_features, input_features)
        self.fc2 = nn.Linear(input_features, 1)

        # Regression Part
        self.fc3 = nn.Linear(input_features + material_features + 1, hidden_dim)  # +1 for size_out[t]
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, 1)

    def forward(self, x, y):
        x0 = self.transformer(x.unsqueeze(1))
        x1 = self.fc1(x0.squeeze())
        x2 = self.fc2(x1)
        size_out = x2.squeeze()

        predict_vector = []
        for t, size in enumerate(size_out.int()):
            if size == 0:
                predict_vector.append([])
                continue

            reg_input = torch.cat([size.view(1), x[t], y[t]])
            reg_out = F.relu(self.fc3(reg_input))
            reg_out = F.relu(self.fc4(reg_out))
            reg_value = self.fc5(reg_out)

            predict_vector.append([reg_value.item()] * int(size.item()))

        return size_out, predict_vector


