# dataset.py

from torch.utils.data import Dataset
import numpy as np

class NumpyDataset(Dataset):
    """
    np.ndarray 형식의 데이터셋

    Attributes
    ----------
    data: np.ndarray
        np.ndarray 형식의 데이터
    label: np.ndarray
        np.ndarray 형식의 레이블
    transfrom: torchvision.transforms.transforms.Compose
        Data transform을 수행하는 클래스
    target_transform: torchvision.transforms.transforms.Compose
        Label transform을 수행하는 클래스


    Methods
    ---------- 
    __len__:
        데이터 길이를 반환
    __getitem__(idx):
        데이터의 idx-th번째 인덱스의 데이터를 반환

    Examples
    ----------
    >>> train_dataset = NumpyDataset(data=train_data, label=train_label, 
                            transform=NpToTensor(), target_transform=NpToTensor())
    """
    def __init__(self, data, label, transform=None, target_transform=None):
        self.data = data
        self.label = label
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = np.array(self.data[idx, :]).astype("float32")
        t = np.array(self.label[idx]).astype("int64")

        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            t = self.target_transform(t)

        return x, t
    