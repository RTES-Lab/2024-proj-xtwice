# databuilder.py

import pandas as pd
import numpy as np

import os

from tqdm import tqdm

from box import Box
from typing import List, Tuple

from sklearn.preprocessing import LabelEncoder


def calculate_statistics(values):
    """
    통계값 계산해주는 함수
    """
    peak = np.max(np.abs(values))
    average = np.mean(values)
    rms = np.sqrt(np.mean(values**2))
    crest_factor = peak / rms
    return peak, average, rms, crest_factor


def make_dataframe(config: Box, rpm: str, directory: str) -> pd.DataFrame:
    """
    주어진 디렉토리에서 각 데이터를 적절히 변환하여 DataFrame을 반환하는 함수

    Parameters
    ----------
    config : Box
        YAML 파일에서 로드한 설정 값을 포함하는 객체
    rpm : str
        필터링할 rpm 값 (예: '1200')
    directory : str
        데이터가 저장된 디렉토리 경로

    Returns
    -------
    pd.DataFrame
        데이터프레임 객체, 각 fault_type에 대한 데이터 및 변환된 값을 포함
    """
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

    # 1105 중 rpm 1200인 데이터만 사용
    filtered_dirs = [
        os.path.join(directory, d) 
        for d in os.listdir(directory) 
        if rpm in d and os.path.isdir(os.path.join(directory, d))
    ]

    # 가장 작은 csv 파일 길이에 맞춤
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

            # 마커 A 기준. 마커 B 사용하고 싶다면,
            # if target_marker == 'B':
            #     data = pd.read_csv(file).iloc[:x_len, 1].values
            data = pd.read_csv(file).iloc[:x_len, 0].values

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


def augment_dataframe(df : pd.DataFrame, target_axis: str, sample_size: int, overlap: int) -> pd.DataFrame:
    """
    주어진 DataFrame에 대해 슬라이딩 윈도우 방식으로 데이터를 증강하는 함수.

    Parameters
    ----------
    df : pd.DataFrame
        원본 데이터프레임으로, 각 샘플에 대한 축(target_axis) 값과 고장 유형(fault_type)을 포함.
    target_axis : str
        데이터프레임에서 증강을 수행할 축을 지정하는 열 이름. x, y, z 중 하나를 선택하면 되지만,
        본 프로젝트는 z축 방향의 데이터를 사용하기로 했으므로 'z'를 입력해야 함.
    sample_size : int
        각 샘플의 크기 (윈도우 크기)
    overlap : int
        겹치는 샘플 수를 지정

    Returns
    -------
    augmented_df : pd.DataFrame
        증강된 데이터프레임으로, 각 샘플을 슬라이딩 윈도우 방식으로 나눈 후 고장 유형(fault_type)과 함께 반환.
    """
    augmented_data = []

    for sample in df[target_axis]:
        augmented_samples = sliding_window_augmentation(sample, window_size=sample_size, overlap=overlap)
        augmented_data.extend(augmented_samples)  

    augmented_df = pd.DataFrame({
        'fault_type': np.repeat(df['fault_type'].values, len(augmented_data) // len(df)),
        target_axis: augmented_data
    })

    return augmented_df


def add_rms_peak(df : pd.DataFrame, target_axis: str) -> pd.DataFrame:
    """
    주어진 데이터프레임에 대해 지정된 축(axis) 값에 대해 RMS(Root Mean Square)와 Peak 값을 계산하여 추가하는 함수.

    Parameters
    ----------
    df : pd.DataFrame
        RMS와 Peak 값을 추가할 데이터프레임
    target_axis : str
        RMS와 Peak 값을 계산할 데이터가 포함된 열 이름. 
        본 프로젝트는 z축 방향의 데이터를 사용하기로 했으므로 'z'를 입력해야 함.

    Returns
    -------
    pd.DataFrame
        RMS와 Peak 값을 계산하여 추가한 새로운 데이터프레임을 반환. 
        기존 데이터프레임에 'rms'와 'peak'라는 열이 추가됨.
    """
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


def get_data_label(df: pd.DataFrame, target: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    주어진 데이터프레임에서 특정 축의 데이터를 추출하고, 레이블을 인코딩하여 반환하는 함수.
    
    Parameters
    ----------
    df : pd.DataFrame
        데이터프레임으로, 'fault_type' 컬럼과 주어진 target 컬럼을 포함.
    target : str
        분석할 데이터가 포함된 컬럼 이름. 해당 컬럼의 값들이 특징값(X)로 사용됨. 'z', 'rms', 'peak' 중 하나여야 함.
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        - X : 특성값이 저장된 NumPy 배열 (target 컬럼의 값들)
        - Y : 레이블값이 저장된 NumPy 배열 ('fault_type' 컬럼의 인코딩된 값)
    """
    label_encoder = LabelEncoder()
    df['fault_type_encoded'] = label_encoder.fit_transform(df['fault_type'])
    Y = df['fault_type_encoded'].values
    
    arr = np.vstack(df[target]) 
    for i in range(len(label_encoder.classes_)):
        print(f"{i}: {label_encoder.classes_[i]}")
    X = np.hstack([arr])

    return X, Y