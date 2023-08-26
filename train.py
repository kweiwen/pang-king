from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import numpy as np
from model import *
from ISMdataSet import *
from tqdm import tqdm

loaded_results = np.load('results.npy', allow_pickle=True)

def collate_fn(batch):
    dimension, tx, rx, order, materials, sequences, lengths = zip(*batch)

    max_len = 32768
    padded_sequences = np.array([sublist + [0] * (max_len - len(sublist)) for sublist in sequences])

    return torch.tensor(np.array(dimension), dtype=torch.float32), \
           torch.tensor(np.array(tx), dtype=torch.float32), \
           torch.tensor(np.array(rx), dtype=torch.float32), \
           torch.tensor(np.array(order), dtype=torch.float32), \
           torch.tensor(np.array(materials), dtype=torch.float32), \
           torch.tensor(padded_sequences, dtype=torch.float32), \
           torch.tensor(np.array(lengths), dtype=torch.long)

def length_loss(input, target):
    loss = torch.abs(target - input) / target
    return loss.mean()


def masked_relative_error_loss(predict, target, predicted_lengths, epsilon=1e-8):
    batch_size, max_seq_len = predict.shape[0], predict.shape[1]

    # Convert predicted_lengths to integer
    predicted_lengths = predicted_lengths.int()

    # Create a mask based on predicted_lengths
    mask = torch.arange(max_seq_len).expand(batch_size, max_seq_len).to(predicted_lengths.device)
    mask = (mask < predicted_lengths.unsqueeze(1)).float()

    # Check for zero predicted lengths
    sum_mask = mask.sum()
    if sum_mask.item() == 0:
        return torch.tensor(2.0, device=predict.device)  # Maximum relative error
    else:
        # get max value
        values, indices = torch.max(target, dim=1)
        values_reshaped = values.view(64, 1)

        # normalization
        target = target / values_reshaped
        predict = predict / values_reshaped

        h_hat = torch.fft.fft(predict)
        h = torch.fft.fft(target)

        amplitude_hat = torch.abs(h_hat)
        amplitude = torch.abs(h)

        phase_hat = torch.angle(h_hat) / torch.pi
        phase = torch.angle(h) / torch.pi

        loss = (amplitude_hat - amplitude) ** 2 + (phase_hat - phase) ** 2
        masked_loss = torch.clamp(loss * mask, 0, 1)

        # Compute the mean relative error
        loss = masked_loss.sum() / sum_mask

        return loss


# train
def train(model, dataloader, optimizer, epochs=32):
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch + 1}/{epochs}")

        for batch_idx, (d, tx, rx, order, materials, sequences, lengths) in progress_bar:
            d, tx, rx, order, materials, sequences, lengths = d.to(device), tx.to(device), rx.to(device), order.to(device), materials.to(device), sequences.to(device), lengths.to(device)

            optimizer.zero_grad()

            predicted_lengths, vector_outs = model(d, tx, rx, order, materials)

            loss1 = length_loss(predicted_lengths, lengths.float())

            # values, indices = torch.max(sequences, dim=1)
            #
            # values_reshaped = values.view(dataloader.batch_size, 1)
            # h_hat = torch.fft.fft(vector_outs/values_reshaped)
            # h = torch.fft.fft(sequences/values_reshaped)
            # amp_hat = torch.abs(h_hat)
            # amp = torch.abs(h)
            # phase_hat = torch.angle(h_hat) / torch.pi
            # phase = torch.angle(h) / torch.pi

            # loss2 = F.l1_loss(phase_hat, phase)
            # loss3 = F.l1_loss(amp_hat, amp)
            # loss = loss1 * 0.5 + loss2 * 0.25 + loss3 * 0.25

            loss = loss1

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            avg_loss = total_loss / (batch_idx + 1)
            progress_bar.set_postfix(AverageLoss=avg_loss, Loss1=loss1.item())



# data processor
d_raw = [x['d'] for x in loaded_results]
d_data = np.array(d_raw)

tx_raw = [x['tx'] for x in loaded_results]
tx_data = np.array(tx_raw)

rx_raw = [x['rx'] for x in loaded_results]
rx_data = np.array(rx_raw)

order_raw = [x['order'] for x in loaded_results]
order_data = np.array(order_raw)

materials_raw = [x['materials_vector'] for x in loaded_results]
materials_vector = np.array(materials_raw)

sequences_data = [x['transform_list'] for x in loaded_results]
lengths_data = np.array([len(sequence) for sequence in sequences_data])

# init data set
dataset = ISMdataSet(d_data, tx_data, rx_data, order_data, materials_vector, sequences_data, lengths_data)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True, collate_fn=collate_fn)

# init model and optimizer
device = torch.device("mps" if torch.backends.mps.is_available() and torch.backends.mps.is_built() else "cpu")
model = CustomModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# execute
train(model, dataloader, optimizer)
