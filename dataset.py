import os
import torch
from torch.utils.data import Dataset

class REMI_dataset(Dataset):
    def __init__(self, root):
        self.root = root
        fs = [os.path.join(root, f) for f in os.listdir(self.root)] 
        self.data_files = [f for f in fs if os.path.isfile(f)]
    def __len__(self):
        return len(self.data_files)
    def __getitem__(self, idx):
        x, y = torch.load(self.data_files[idx])
        return x.to(torch.int64), y.to(torch.int64)