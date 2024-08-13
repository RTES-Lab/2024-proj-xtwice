# databuilder.py

import os
import pandas as pd
from box import Box 
import numpy as np
from typing import Tuple

def make_dataframe(root: str, config: Box):
    """
    csv 파일로부터 데이터를 읽어 데이터프레임을 생성하는 함수

    Parameters
    ---------- 
    root: str
        output 파일의 루트 디렉토리
    config: Box

    Returns
    ----------
    pd.DataFrame
        'Adata', 'Bdata', 'fault_type', 'axis', 'label'을 컬럼으로 갖는 데이터프레임
        
    Examples
    ----------
    >>> from box import Box
    >>> config = load_yaml('./model_config.yaml') # Box 객체를 반환하는 함수 예시
    >>> df = make_dataframe('../output', config)
    """
    df = {}
    df['Adata'] = []
    df['Bdata'] = []
    df['fault_type'] = []
    df['axis'] = []
    df['label'] = []

    for fault_type, label in config.label_map.items():
        fault_dir = os.path.join(root, f'{config.date}_{config.bearing_type}_{fault_type}_{config.rpm}')
        for axis in config.csv_files:
            fault_file = os.path.join(fault_dir, axis)
            data = pd.read_csv(fault_file, header=None)
            Adata = data.iloc[:config.max_len, 0].values.flatten()  
            Bdata = data.iloc[:config.max_len, 1].values.flatten() 
            df['Adata'].append(Adata)
            df['Bdata'].append(Bdata)
            df['fault_type'].append(fault_type)
            df['axis'].append(axis[0])
            df['label'].append(label)
    df = pd.DataFrame(df)

    # TODO normalize 방식 변경 필요
    for data in ['Adata', 'Bdata']:
        df[data] = df[data].apply(lambda x: x - np.mean(x))

    return df

def data_sampling(
        data_segment: np.ndarray, sample_length: int, shift: int, 
        num_classes:int, classID:int, is_onehot: bool = False
        )-> Tuple[np.ndarray, np.ndarray]:
    """
    데이터 segment로부터 데이터, 레이블을 샘플링하는 함수

    Parameters
    ---------- 
    data: np.ndarray
        input 데이터 segment
    sample_length: int
        input 데이터 샘플의 길이
    shift: int
        overlapping을 사용할 때 각 샘플간 interval. sample_length=shift라면 overlapping이 없다.
    num_classes: int
        클래스 개수
    classID
        데이터의 클래스 id
    is_onehot: bool
        원핫인코딩 유무

    Returns
    ----------
    Tuple[np.ndarray, np.ndarray]
        튜플 (데이터, 라벨)
    """
    sampled_data = np.array([
        # data segment에서 sample_length만큼의 데이터를 shift만큼 이동해가며 np.ndarray 형태로 반환
        data_segment[i: i+sample_length]
        for i in range(0, len(data_segment)-sample_length, shift)
    ])
    if is_onehot: # 원 핫 인코딩
        label = np.zeros((sampled_data.shape[0], num_classes))
        label[:, classID] = 1
    else: # 레이블 인코딩
        label = np.zeros((sampled_data.shape[0]))
        label = label + classID
    
    return sampled_data, label

def build_from_dataframe(
        df: pd.DataFrame, sample_length: int, shift: int, data_column: str,
        is_onehot: bool = False
        )-> Tuple[np.ndarray, np.ndarray]:
    """
    데이터프레임으로부터 np.ndarray 타입의 (데이터, 라벨) 튜플을 만드는 함수

    Parameters
    ---------- 
    df: pd.DataFrame
        input 데이터프레임
    sample_length: int
        input 데이터 샘플의 길이
    shift: int
        overlapping을 사용할 때 각 샘플간 interval
    data_column: str
        사용할 data column
    is_onehot: bool
        원핫인코딩 유무

    Returns
    ----------
    Tuple[np.ndarray, np.ndarray]
        튜플 (데이터, 라벨)
    """
    
    num_classes = df["label"].max() - df["label"].min() + 1
    num_data = len(df)

    data = []
    label = []

    for i in range(num_data):
        data_segment = df.iloc[i][data_column]

        dataseg, labelseg = data_sampling(data_segment, sample_length, shift,
                                            num_classes, df.iloc[i]["label"], is_onehot)
                                            
        
        data.append(dataseg)
        label.append(labelseg)

    data_array = np.concatenate(tuple(data), axis=0)
    label_array = np.concatenate(tuple(label), axis=0)

    return data_array, label_array