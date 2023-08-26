from torch.utils.data import Dataset
class ISMdataSet(Dataset):
    def __init__(self, dimension, tx, rx, order, materials, sequences, lengths):
        self.dimension = dimension
        self.tx = tx
        self.rx = rx
        self.order = order
        self.materials = materials
        self.sequences = sequences
        self.lengths = lengths

    def __len__(self):
        return len(self.dimension)

    def __getitem__(self, idx):
        return self.dimension[idx], self.tx[idx], self.rx[idx], self.order[idx], self.materials[idx], self.sequences[idx], self.lengths[idx]
