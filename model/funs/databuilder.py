# databuilder.py

import pandas as pd
import numpy as np

import os

from tqdm import tqdm

from typing import List, Tuple

from sklearn.preprocessing import LabelEncoder

def calculate_statistics(values):
    peak = np.max(np.abs(values))
    average = np.mean(values)
    rms = np.sqrt(np.mean(values**2))
    crest_factor = peak / rms
    return peak, average, rms, crest_factor

def make_dataframe(config, rpm, directory) -> pd.DataFrame:

    df = {}
    df['dir_name'] = []
    df['fault_type'] = []
    df['x'] = []
    df['z'] = []
    df['label'] = []

    label_dic = {
        'H'  : 0,
        'B'  : 1,
        'IR' : 2,
        'OR' : 3
    }

    filtered_dirs = [
        os.path.join(directory, d) 
        for d in os.listdir(directory) 
        if rpm in d and os.path.isdir(os.path.join(directory, d))
    ]

    x_len = 66420
    conversion_factors = config.conversion_factors

    for sub_dirs in tqdm(sorted(filtered_dirs)):
        parts = sub_dirs.split('/')[-1]
        parts = parts.split('_')
        date = parts[0]
        view = parts[-1]
        fault_type = parts[-2]
        bearing_type = parts[-4]

        df['fault_type'].append(fault_type)
        df['label'].append(label_dic[fault_type])
        df['dir_name'].append(sub_dirs)

        axis_list = config.axis_to_csv_dic[view]

        for axis_csv in axis_list:
            axis = axis_csv[0]
            file = os.path.join(sub_dirs, axis_csv)

            # marker A
            data = pd.read_csv(file).iloc[:x_len, 0].values
            # if target_marker == 'B':
            #     data = pd.read_csv(file).iloc[:x_len, 1].values

            conversion_factor = conversion_factors.get(date, {}).get(view, 1).get(fault_type, 1)

            data = data * conversion_factor
            df[axis].append(np.array(data))

    return pd.DataFrame(df)


def sliding_window_augmentation(data, window_size=2048, overlap=1024):
    """
    데이터에 슬라이딩 윈도우를 적용하여 증강된 샘플 생성.
    
    Args:
        data (np.ndarray): 1D 배열 데이터.
        window_size (int): 윈도우 크기.
        overlap (int): 윈도우 간 겹치는 크기.

    Returns:
        np.ndarray: 슬라이딩 윈도우로 분할된 데이터 배열 (2D 배열).
    """
    step_size = window_size - overlap  # 윈도우가 한 번에 이동하는 크기
    augmented_data = []

    for start in range(0, len(data) - window_size + 1, step_size):
        window = data[start:start + window_size]
        augmented_data.append(window)

    return np.array(augmented_data)

def augment_dataframe(df : pd.DataFrame, axis: str, sample_size: int, overlap: int):
    rms_values = []
    peak_values = []
    augmented_data = []

    for sample in df[axis]:
        augmented_samples = sliding_window_augmentation(sample, window_size=sample_size, overlap=overlap)
        augmented_data.extend(augmented_samples)  # 모든 윈도우를 누적

    # 증강된 데이터로 새로운 DataFrame 생성
    augmented_df = pd.DataFrame({
        'fault_type': np.repeat(df['fault_type'].values, len(augmented_data) // len(df)),
        axis: augmented_data
    })

    return augmented_df

def add_rms_peak(df : pd.DataFrame, target_axis):
    rms_values = []
    peak_values = []

    for sample in df[target_axis]:
        peak, average, rms, crest_factor = calculate_statistics(sample)
        rms_values.append(rms)
        peak_values.append(peak)

    # 데이터프레임에 RMS와 Peak 열 추가
    df['rms'] = rms_values
    df['peak'] = peak_values

    return df

def get_data_label(df, target):
    label_encoder = LabelEncoder()
    df['fault_type_encoded'] = label_encoder.fit_transform(df['fault_type'])
    Y = df['fault_type_encoded'].values

    arr = np.vstack(df[target]) 

    X = np.hstack([arr])

    return X, Y