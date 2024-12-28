import torch
import torch.utils
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, data, label): 
        self.data = data
        self.label = label

    def __getitem__(self, index):
        data = torch.tensor(self.data[index]).float()
        label = torch.tensor(self.label[index]).int()
        return data, label
    
    def __len__(self):
        return self.data.shape[0]
    
def get_dataloader(dataset: Dataset, batch_size: int, shuffle: bool = False) -> DataLoader:
    return DataLoader(dataset, batch_size, shuffle)
