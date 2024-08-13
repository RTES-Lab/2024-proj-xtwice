# utils.py

import numpy as np
import torch

import random

import yaml
from box import Box

import matplotlib.pyplot as plt

def load_yaml(config_yaml_file: str):
    """
    YAML 파일을 읽어와 Box 객체로 변환하는 함수.

    Parameters
    ----------
    config_yaml_file : str
        읽을 YAML 파일의 경로.

    Returns
    ----------
    config : Box
        YAML 파일의 내용을 포함한 Box 객체
    """
    with open(config_yaml_file) as f:
        config_yaml = yaml.load(f, Loader=yaml.FullLoader)
        config = Box(config_yaml)
    return config

def set_seed(seed: int):
    """
    랜덤 시드 고정 함수.

    Parameters
    ----------
    seed : int
        고정할 시드 값
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # CUDA를 사용하는 경우
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def plot_training_curves(train_loss_list, train_acc_list, data):
    epochs = range(1, len(train_loss_list) + 1)  

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss_list, label='Train Loss', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Train Loss')
    plt.title(f'{data} Train Loss over Epochs')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc_list, label='Train Accuracy', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Train Accuracy')
    plt.title(f'{data} Train Accuracy over Epochs')
    plt.grid(True)
    plt.legend()

    # plt.tight_layout()
    plt.savefig(f'./result_{data}.png')