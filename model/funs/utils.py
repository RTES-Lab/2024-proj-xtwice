# utils.py

import yaml
from box import Box

import numpy as np
import pandas as pd
import random

import torch
import tensorflow as tf

import matplotlib.pyplot as plt

def load_yaml(config_path: str) -> Box:
    """
    YAML 파일을 load하는 함수

    Parameters
    ----------
    config_path : str
        YAML 파일 경로

    Returns
    -------
    Box 
        Box 개체
    """
    with open(config_path) as f:
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
    tf.random.set_seed(seed) 
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_peak_rms_hist_pic(df: pd.DataFrame, save_path: str):
    """
    주어진 데이터프레임에서 각 fault_type에 대한 Peak과 RMS 값의 분포를 히스토그램으로 생성하고 저장하는 함수

    Parameters
    ----------
    df : pd.DataFrame
        fault_type, peak, rms 등의 값이 포함된 데이터프레임
    save_path : str
        히스토그램 이미지를 저장할 파일 경로
    """
    fault_type_list = ['H', 'OR', 'B', 'IR']
    
    plt.figure(figsize=(20, 15))
    
    # Peak 히스토그램 (첫 번째 subplot)
    plt.subplot(2, 1, 1)
    for fault_type in fault_type_list:
        subset = df[df['fault_type'] == fault_type]
        plt.hist(subset['peak'], bins=50, alpha=0.5, label=fault_type)
    plt.title('Peak Distribution', size=50)  
    plt.xlabel('Peak [mm]', size=40)  
    plt.ylabel('Counts', size=40)  
    plt.xticks(fontsize=30)  
    plt.yticks(fontsize=30)  
    plt.legend(fontsize=18, loc='upper right') 

    # RMS 히스토그램 (두 번째 subplot)
    plt.subplot(2, 1, 2)
    for fault_type in fault_type_list:
        subset = df[df['fault_type'] == fault_type]
        plt.hist(subset['rms'], bins=50, alpha=0.5, label=fault_type)
    plt.title('RMS Distribution', size=50)  
    plt.xlabel('RMS [mm]', size=40)  
    plt.ylabel('Counts', size=40)  
    plt.xticks(fontsize=30)  
    plt.yticks(fontsize=30)  
    plt.legend(fontsize=18, loc='upper right') 
    
    # 레이아웃 조정 및 저장
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"히스토그램이 {save_path}에 저장되었습니다.")