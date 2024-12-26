# databuilder.py

import pandas as pd
import numpy as np

import os

from tqdm import tqdm

from box import Box
from typing import List, Tuple, Optional

from sklearn.preprocessing import LabelEncoder, StandardScaler

def make_dataframe(
        config: Box, directory_list: List[str], target_marker: Optional[str] = 'A', max_len: Optional[int] = None
        ) -> pd.DataFrame:
    """
    주어진 디렉토리에서 각 데이터를 적절히 변환하여 DataFrame을 반환하는 함수


    Parameters
    ----------
    config : Box
        YAML 파일에서 로드한 설정 값을 포함하는 객체
    directory_list : List[str]
        데이터가 저장된 디렉토리 경로 리스트
    target_marker : Optional[str], optional
        타겟 마커. 기본값은 'A'
    max_len : Optional[int], optional
        데이터를 읽을 최대 길이. None이면 모든 데이터를 읽어온다. 기본값은 None

    Returns
    -------
    pd.DataFrame
        데이터프레임 객체, 각 fault_type에 대한 축 별 변위 데이터를 포함
    """
    df = {}
    df['dir_name'] = []
    df['fault_type'] = []
    df['label'] = []

    conversion_factors = config.conversion_factors
    label_dic = config.label_dic

    for sub_dirs in tqdm(sorted(directory_list)):
        parts = sub_dirs.split(os.path.sep)[-1]
        parts = parts.split('_')

        date = parts[0]
        bearing = parts[1]
        rpm = int(parts[2])
        fault_type = parts[3]
        view = parts[4]

        df['fault_type'].append(fault_type)
        df['label'].append(label_dic[fault_type])
        df['dir_name'].append(sub_dirs)

        axis_list = config.axis_to_csv_dic[view]
        for axis_csv in axis_list:
            axis = axis_csv[0]  
            if axis not in df:
                df[axis] = []  

        for axis_csv in axis_list:
            axis = axis_csv[0]
            file = os.path.join(sub_dirs, axis_csv)

            if target_marker == 'A':
                data = pd.read_csv(file).iloc[:max_len, 0].values
            if target_marker == 'B':
                data = pd.read_csv(file).iloc[:max_len, 1].values

            if int(date) < 1100:
                conversion_factor = conversion_factors.get(date, {}).get(view, 1)
            elif int(date) == 1105 or int(date) == 1217:
                conversion_factor = conversion_factors.get(date, {}).get(view, 1).get(fault_type, 1)
            else:
                conversion_factor = conversion_factors.get(date, {}).get(view, 1).get(target_marker, 1)

            data = data * conversion_factor
            data -= np.mean(data)

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

def normalize_data(data: np.ndarray) -> np.ndarray:
    """
    주어진 데이터를 정규화하는 함수. (Standardization)
    
    Parameters
    ----------
    data : np.ndarray
        정규화할 데이터 (2D numpy array 형태)

    Returns
    -------
    np.ndarray
        정규화된 데이터
    """
    # 각 열의 평균과 표준편차 계산
    means = np.mean(data, axis=0)  # 각 열의 평균
    std_devs = np.std(data, axis=0)  # 각 열의 표준편차
    
    # 표준화
    standardized_data = (data - means) / std_devs
    return standardized_data

def add_statistics(df: pd.DataFrame, target_axis: List[str]) -> pd.DataFrame:
    """
    통계값을 계산한 후 Standardization을 적용하여 데이터프레임에 추가하는 함수.

    Parameters
    ----------
    df : pd.DataFrame
        원본 데이터프레임.
    target_axis : list
        통계 값을 계산할 대상 축의 이름 리스트. e.g., ['x', 'z']

    Returns
    -------
    pd.DataFrame
        Standardization된 통계값이 추가된 데이터프레임.
    """
    for axis in target_axis:
        # 각 축에 대해 통계값 저장용 리스트 초기화
        rms_values = []
        peak_values = []
        skewness_values = []
        kurtosis_values = []
        fused_feature_values = []  

        # 축 데이터가 존재하는지 확인
        if axis not in df.columns:
            raise ValueError(f"Target axis '{axis}' not found in DataFrame columns.")

        # 통계 값 계산
        for sample in df[axis]:
            peak, _, rms, _, _, skew, kurt = calculate_statistics(sample)
            rms_values.append(rms)
            peak_values.append(peak)
            skewness_values.append(skew)
            kurtosis_values.append(kurt)
            fused_feature_values.append([rms, skew, kurt]) 

        # 임시 데이터프레임 생성 (정규화를 위한)
        temp_df = pd.DataFrame({
            f'{axis}_rms': rms_values,
            f'{axis}_skewness': skewness_values,
            f'{axis}_kurtosis': kurtosis_values,
            f'{axis}_peak': peak_values
        })

        # numpy 배열로 변환 후 정규화
        temp_np_array = temp_df.to_numpy()
        standardized_data = normalize_data(temp_np_array)

        # 정규화된 값만 데이터프레임에 추가
        df[f'{axis}_rms'] = standardized_data[:, 0]
        df[f'{axis}_skewness'] = standardized_data[:, 1]
        df[f'{axis}_kurtosis'] = standardized_data[:, 2]
        df[f'{axis}_peak'] = standardized_data[:, 3]

        # fused_feature값도 정규화된 값으로 갱신
        for i in range(len(df)):
            fused_feature_values[i] = list(standardized_data[i, :])

        df[f'{axis}_fused_features'] = fused_feature_values

    return df



def augment_dataframe(
        df: pd.DataFrame, target_axes: list, sample_size: int = 2048, overlap: int = 1024
        ) -> pd.DataFrame:
    """
    주어진 데이터프레임에서 여러 축(target_axes)의 데이터를 슬라이딩 윈도우 기법으로 증강하여 새로운 데이터프레임을 생성

    Parameters
    ----------
    df : pd.DataFrame
        원본 데이터프레임. 고장 유형('fault_type') 및 여러 축 데이터(target_axes)를 포함해야 함.
    target_axes : list
        증강을 수행할 데이터 축의 이름 리스트. 예: ['x', 'z']
    sample_size : int, optional
        슬라이딩 윈도우의 크기(샘플 단위), 기본값은 2048.
    overlap : int, optional
        슬라이딩 윈도우의 오버랩 크기(샘플 단위), 기본값은 1024.

    Returns
    -------
    pd.DataFrame
        증강된 데이터프레임. 각 고장 유형과 여러 증강된 축 데이터가 포함됨.
    """
    augmented_data = {axis: [] for axis in target_axes}  
    augmented_fault_types = [] 

    for i in range(len(df)):
        fault_type = df['fault_type'].iloc[i]
        num_windows = None 

        for axis in target_axes:
            sample = df[axis].iloc[i]
            augmented_samples = sliding_window_augmentation(
                sample, window_size=sample_size, overlap=overlap
            )
            augmented_data[axis].extend(augmented_samples)

            if num_windows is None:
                num_windows = len(augmented_samples)
            elif num_windows != len(augmented_samples):
                raise ValueError("All axes must produce the same number of windows.")

        augmented_fault_types.extend([fault_type] * num_windows)

    augmented_df = {'fault_type': augmented_fault_types}
    for axis, data in augmented_data.items():
        augmented_df[axis] = data

    return pd.DataFrame(augmented_df)


def calculate_statistics(values):
    """
    통계값을 계산해주는 함수
    """
    peak = np.max(np.abs(values))
    average = np.mean(values)
    rms = np.sqrt(np.mean(values**2))
    crest_factor = peak / rms

    max_value = np.max(values)
    minvalue = np.min(values)
    pk2pk = max_value - minvalue

    s_dividend = np.mean(np.power(values-average, 3))
    s_divisor = np.power(np.mean(np.power(values-average, 2)), 3/2)
    skew = s_dividend / s_divisor

    k_dividend = np.mean(np.power(values-average, 4))
    k_divisor = np.power(np.mean(np.power(values-average, 2)), 2)
    kurt = k_dividend / k_divisor

    return peak, average, rms, crest_factor, pk2pk, skew, kurt

def add_statistics(df: pd.DataFrame, target_axis: list) -> pd.DataFrame:
    """
    통계값을 계산한 후 Standardization을 적용하여 데이터프레임에 추가하는 함수.

    Parameters
    ----------
    df : pd.DataFrame
        원본 데이터프레임.
    target_axis : list
        통계 값을 계산할 대상 축의 이름 리스트. e.g., ['x', 'z']

    Returns
    -------
    pd.DataFrame
        Standardization된 통계값이 추가된 데이터프레임.
    """
    for axis in target_axis:
        # 각 축에 대해 통계값 저장용 리스트 초기화
        rms_values = []
        peak_values = []
        skewness_values = []
        kurtosis_values = []
        fused_feature_values = []  

        # 축 데이터가 존재하는지 확인
        if axis not in df.columns:
            raise ValueError(f"Target axis '{axis}' not found in DataFrame columns.")

        # 통계 값 계산
        for sample in df[axis]:
            peak, _, rms, _, _, skew, kurt = calculate_statistics(sample)
            rms_values.append(rms)
            peak_values.append(peak)
            skewness_values.append(skew)
            kurtosis_values.append(kurt)
            fused_feature_values.append([rms, skew, kurt]) 

        # 임시 데이터프레임 생성 (정규화를 위한)
        temp_df = pd.DataFrame({
            f'{axis}_rms': rms_values,
            f'{axis}_skewness': skewness_values,
            f'{axis}_kurtosis': kurtosis_values,
        })

        # 정규화 (Standardization)
        scaler = StandardScaler()
        standardized_data = scaler.fit_transform(temp_df)

        # 정규화된 값만 데이터프레임에 추가
        df[f'{axis}_rms'] = standardized_data[:, 0]
        df[f'{axis}_skewness'] = standardized_data[:, 1]
        df[f'{axis}_kurtosis'] = standardized_data[:, 2]

        for i in range(len(df)):
            fused_feature_values[i] = list(standardized_data[i, :])

        df[f'{axis}_fused_features'] = fused_feature_values

    return df

# def add_statistics(df: pd.DataFrame, target_axis: List[str]) -> pd.DataFrame:
    # """
    # 주어진 데이터프레임에서 특정 축에 대해 통계적 특성을 계산하여
    # 이를 데이터프레임에 새로운 열로 추가하는 함수.

    # Parameters
    # ----------
    # df : pd.DataFrame
    #     통계 값을 추가할 원본 데이터프레임. 반드시 target_axis에 해당하는 데이터가 포함되어야 함.
    # target_axis : list
    #     통계 값을 계산할 대상 축의 이름 리스트. e.g. ['x', 'z']

    # Returns
    # -------
    # pd.DataFrame
    #     통계 값이 추가된 데이터프레임. 
    # """
    # for axis in target_axis:
    #     # 각 축에 대해 결과 리스트 초기화
    #     rms_values = []
    #     peak_values = []
    #     skewness_values = []
    #     kurtosis_values = []
    #     fused_feature_values = []

    #     # 축 데이터가 존재하는지 확인
    #     if axis not in df.columns:
    #         raise ValueError(f"Target axis '{axis}' not found in DataFrame columns.")

    #     # 통계 값 계산
    #     for sample in df[axis]:
    #         peak, _, rms, _, _, skew, kurt = calculate_statistics(sample)
    #         rms_values.append(rms)
    #         peak_values.append(peak)
    #         skewness_values.append(skew)
    #         kurtosis_values.append(kurt)
    #         fused_feature_values.append([rms, skew, kurt])


    #     # 통계 값 추가
    #     df[f'{axis}_rms'] = rms_values
    #     df[f'{axis}_peak'] = peak_values
    #     df[f'{axis}_skewness'] = skewness_values
    #     df[f'{axis}_kurtosis'] = kurtosis_values
    #     df[f'{axis}_fused_features'] = fused_feature_values

    # return df

def get_data_label(df: pd.DataFrame, target: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    주어진 데이터프레임에서 특정 축의 데이터를 추출하고, 레이블을 인코딩하여 반환하는 함수.
    
    Parameters
    ----------
    df : pd.DataFrame
        데이터프레임으로, 'fault_type' 컬럼과 주어진 target 컬럼을 포함.
    target : str
        분석할 데이터가 포함된 컬럼 이름. 해당 컬럼의 값들이 특징값(X)로 사용됨.
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        - X : 특성값이 저장된 NumPy 배열 (target 컬럼의 값들)
        - Y : 레이블값이 저장된 NumPy 배열 ('fault_type' 컬럼의 인코딩된 값)
    """
    label_encoder = LabelEncoder()
    df['fault_type_encoded'] = label_encoder.fit_transform(df['fault_type'])

    # print("클래스 이름과 레이블 매핑:")
    # for class_name, label in zip(label_encoder.classes_, range(len(label_encoder.classes_))):
    #     print(f"{class_name}: {label}")
    
    Y = df['fault_type_encoded'].values
    
    arr = np.vstack(df[target]) 
    X = np.hstack([arr])

    return X, Y