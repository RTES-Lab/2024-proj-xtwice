# processing.py

import torch

class NpToTensor:
    """numpy array를 텐서로 변환"""
    def __call__(self, x):
        return torch.from_numpy(x)
    