from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import numpy as np
from model import *
from ISMdataSet import *

loaded_results = np.load('results.npy', allow_pickle=True)

def collate_fn(batch):
    inputs, materials, sequences, lengths = zip(*batch)

    max_len = 32768
    padded_sequences = np.array([sublist + [0] * (max_len - len(sublist)) for sublist in sequences])

    return torch.tensor(np.array(inputs), dtype=torch.float32), \
           torch.tensor(np.array(materials), dtype=torch.float32), \
           torch.tensor(padded_sequences, dtype=torch.float32), \
           torch.tensor(np.array(lengths), dtype=torch.long)

def length_loss(input, target):
    loss = torch.abs(target - input) / target
    return loss.mean()

def sample_loss(input, target):
    loss = torch.abs(target - input) ** 2
    return loss.mean()

# data processor
input_raw = [x['input_vector'] for x in loaded_results]
inputs_data = np.array(input_raw)

materials_raw = [x['materials_vector'] for x in loaded_results]
materials_vector = np.array(materials_raw)

sequences_data = [x['transform_list'] for x in loaded_results]

lengths_data = np.array([len(sequence) for sequence in sequences_data])

dataset = ISMdataSet(inputs_data, materials_vector, sequences_data, lengths_data)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)

# init model and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CustomModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# train
def train(model, dataloader, optimizer, epochs=32):
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        for inputs, materials, sequences, lengths in dataloader:
            inputs, materials, sequences, lengths = inputs.to(device), materials.to(device), sequences.to(device), lengths.to(device)

            optimizer.zero_grad()

            predicted_lengths, vector_outs = model(inputs, materials)

            loss1 = length_loss(predicted_lengths, lengths.float())
            # loss2 = sample_loss(vector_outs, sequences)
            # loss = loss1 * 0.5 + loss2 * 0.5
            loss = loss1

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(predicted_lengths, lengths.float())
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}")

# execute
train(model, dataloader, optimizer)
