from torch.utils.data import Dataset
class ISMdataSet(Dataset):
    def __init__(self, inputs, materials, sequences, lengths):
        self.inputs = inputs
        self.materials = materials
        self.sequences = sequences
        self.lengths = lengths

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.materials[idx], self.sequences[idx], self.lengths[idx]
