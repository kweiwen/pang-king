import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


# class CustomModel(nn.Module):
#     def __init__(self, input_features=50, hidden_dim=256):
#         super(CustomModel, self).__init__()
#
#         # Transformer Part
#         encoder_layers = TransformerEncoderLayer(d_model=input_features, nhead=1)
#         self.transformer = TransformerEncoder(encoder_layers, num_layers=2)
#
#         self.length_predictor = nn.Linear(input_features, 1)  # Predict length as an int
#
#         # Regressor Part
#         self.regressor = nn.Linear(1, hidden_dim)  # Regressor will output a vector of length hidden_dim
#
#     def forward(self, x):
#         # Assuming x is of shape (batch_size, sequence_length, input_features)
#         x = x.unsqueeze(1)
#         transformer_out = self.transformer(x)
#
#         # We use the last element of the sequence for length prediction
#         predicted_length = self.length_predictor(transformer_out[:, -1]).squeeze().int()
#
#         # Generate vector based on predicted length
#         vector_out = self.regressor(predicted_length.float().unsqueeze(-1))
#
#         return predicted_length, vector_out

class CustomModel(nn.Module):
    def __init__(self, input_features=14, material=36, hidden_dim=256):
        super(CustomModel, self).__init__()

        # Transformer Part
        encoder_layers = TransformerEncoderLayer(d_model=input_features, nhead=2)
        self.transformer = TransformerEncoder(encoder_layers, num_layers=2)

        self.length_predictor = nn.Linear(input_features, 1)  # Predict length as an int

        # Regressor Part
        self.regressor = nn.Linear(1, hidden_dim)  # Adjusting input size

    def forward(self, x):
        # Assuming x is of shape (batch_size, sequence_length, input_features)
        x = x.unsqueeze(1)
        transformer_out = self.transformer(x)

        # We use the last element of the sequence for length prediction
        predicted_length = self.length_predictor(transformer_out[:, -1]).squeeze()

        # Generate vector based on predicted length
        vector_out = self.regressor(predicted_length.unsqueeze(-1))

        return predicted_length, vector_out


