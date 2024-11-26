import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from typing import Tuple, List, Optional

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import glob


from utils import load_yaml

# config 파일 경로 지정
config_dir = '../' 
config_path = os.path.join(config_dir, 'process_and_extract_config.yaml')
config = load_yaml(config_path)

def drift_correction(df, folder_list):
    time = np.arange(len(df))
    axis = folder_list[-1]
    # if axis == 'T':
    #     p_a = np.polyfit(time, df.iloc[:, 0], 1)
    #     p_b = np.polyfit(time, df.iloc[:, 1], 1)
    #     p_c = np.polyfit(time, df.iloc[:, 2], 1)
    #     p_d = np.polyfit(time, df.iloc[:, 3], 1)
    #     p_e = np.polyfit(time, df.iloc[:, 4], 1)

    #     drift_trend_a = np.polyval(p_a, time)
    #     drift_trend_b = np.polyval(p_b, time)
    #     drift_trend_c = np.polyval(p_c, time)
    #     drift_trend_d = np.polyval(p_d, time)
    #     drift_trend_e = np.polyval(p_e, time)

    #     corrected_a = df.iloc[:, 0] - drift_trend_a
    #     corrected_b = df.iloc[:, 1] - drift_trend_b
    #     corrected_c = df.iloc[:, 2] - drift_trend_c
    #     corrected_d = df.iloc[:, 3] - drift_trend_d
    #     corrected_e = df.iloc[:, 4] - drift_trend_e

    #     corrected_df = pd.DataFrame({'A': corrected_a, 'B': corrected_b, 'C': corrected_c, 'D': corrected_d, 'E': corrected_e})
    # else:
    p_a = np.polyfit(time, df.iloc[:, 0], 1)
    p_b = np.polyfit(time, df.iloc[:, 1], 1)

    drift_trend_a = np.polyval(p_a, time)
    drift_trend_b = np.polyval(p_b, time)

    corrected_a = df.iloc[:, 0] - drift_trend_a
    corrected_b = df.iloc[:, 1] - drift_trend_b
    
    corrected_df = pd.DataFrame({'A': corrected_a, 'B': corrected_b})
    
    return corrected_df


def get_dir_list(
        target_dir: str, 
        target_rpm: Optional[str] = None, target_fault_type: Optional[str] = None, target_view: Optional[str] = None, 
        ):
    '''
    주어진 조건에 맞는 디렉토리 리스트를 반환하는 함수
    
    Parameters
    ----------
    target_dir: str
        검색할 상위 디렉토리 경로.
    target_rpm: str, optional
        디렉토리 이름에 포함될 rpm 유형 정보. '1200', '600'을 입력할 수 있다. 기본값은 None.
    target_fault_type: str, optional
        디렉토리 이름에 포함될 결함 유형 정보. 'H', 'B', 'IR', 'OR'을 입력할 수 있다. 기본값은 None.
    target_view: str, optional
        디렉토리 이름에 포함될 뷰 정보. 'F', 'S', 'T'를 입력할 수 있으며,
        각각은 Front view, Side view, Top view를 나타낸다. 기본값은 None.
    

    Returns
    ----------
    dir_list: List[str]  
        조건에 맞는 디렉토리 경로 리스트.

    동작 설명
    ----------
        - target_view와 target_fault_type이 모두 None이면, target_dir 안의 모든 디렉토리를 반환.
        - target_view만 None이 아니면, 디렉토리 이름에 target_view가 포함된 디렉토리만 반환.
        - target_fault_type만 None이 아니면, 디렉토리 이름에 target_fault_type이 포함된 디렉토리만 반환.
        - target_view와 target_fault_type이 모두 주어지면, 두 값이 모두 포함된 디렉토리만 반환.
    '''
    pattern = os.path.join(target_dir, '*')

    # 정상적인 값이 들어왔는지 체크
    check_param(target_view=target_view, target_fault_type=target_fault_type)

    if target_rpm:
        pattern += f'{target_rpm}*'
    if target_fault_type:
        pattern += f'{target_fault_type}*'
    if target_view:
        pattern += f'{target_view}'
    dir_list = glob.glob(pattern)

    if not dir_list:
        print('The directory that meets your criteria does not exist. No graph will be generated.')

    return sorted(dir_list)
    

def get_data(folder_list: list, csv_file: str, columns: list) -> List[Tuple[str, pd.DataFrame]]:
    """
    데이터를 얻어오는 함수
    """
    data = []
    file_path = os.path.join(folder_list, csv_file)

    df = pd.read_csv(file_path, usecols=list(range(0, len(columns))), names=columns, header=0) 


    # 즉 변위 추출 자체는 start가 빼진 csv인데, 거기서 한 번 더 전처리가 여기서 진행됨!
    #means = df.mean()
    #df_centered = df - means
    #corrected_df = drift_correction(df_centered, folder_list)


    data.append((file_path, df))
            
    return data

def draw_data(data: list, csv_file: str, column: str, axis: str, length: int=None):
    """
    데이터를 그래프로 그리는 함수.
    data: 각 디렉토리별 파일과 데이터가 들어있는 리스트
    csv_file: CSV 파일명
    column: 데이터 열 이름
    axis: 'X', 'Y', 'Z' 축
    directory: 현재 디렉토리명
    """

    view_mapping = {
        'F': 'Front',
        'S': 'Side',
        'T': 'Top'
    }

    conversion_factors = config.conversion_factors

    plt.figure(figsize=(15, 6))

    # 데이터 길이를 맞추기 위한 최소 길이
    if not length:
        min_len = min(len(df) for _, df in data)
    else:
        min_len = length

    for subdir, df in data:
        subdir = os.path.basename(os.path.dirname(subdir))
        # 베어링 타입 추출 ('OR', 'H', 'IR', 'B')
        fault_type = next((x for x in ['OR', 'H', 'IR', 'B'] if x in subdir), subdir)
        df = df[:min_len].copy()

        # 중앙값 기준으로 10배가 넘는 값은 NaN으로 처리(여기서 Outlier에 대한 처리를 할 수 있을 듯?)
        median_value = df[column].abs().median()
        df.loc[df[column].abs() > 10 * median_value, column] = np.nan

        # 시간 축으로 나누고, 변위 데이터는 단위 변환하여 그리기
        # view 별로 marker size에 따른 다르게 그리는 법을 여기서 작성해야 함!

        # 폴더명에서 date와 axis 추출
        parts = subdir.split('_')
        date = parts[0]
        axis = parts[-1]

        # 변환 값 가져오기
        if int(date) < 1100:
            conversion_factor = conversion_factors.get(date, {}).get(axis, 1)
        elif int(date) == 1105:
            conversion_factor = conversion_factors.get(date, {}).get(axis, 1).get(fault_type, 1)
        else:
            conversion_factor = conversion_factors.get(date, {}).get(axis, 1).get(column, 1)
        
        # 시간 축으로 나누고, 변위 데이터는 단위 변환하여 그리기
        plt.plot(df.index / 240, df[column] * conversion_factor, label=fault_type, alpha=0.7)
        #plt.ylim(-0.06, 0.06)
        
        bearing_type = parts[1]
        RPM = parts[2]
    
    description = f'{bearing_type} {fault_type}, {RPM} RPM, {view_mapping[axis]} view, {csv_file[:1]} axis, marker {column}'
    
    plt.title(description, size=30)
    plt.xlabel('Time [s]', size=30)
    plt.ylabel('Displacement [mm]', size=30)
    plt.xticks(fontsize=30)  # x축 틱 글씨 크기 설정
    plt.yticks(fontsize=30)  # y축 틱 글씨 크기 설정
    plt.legend()

    # 그래프 보여주기
    plt.show()
    plt.close()

def draw_single_graphs(dir_list: List, target_csv_list: Optional[List]=None, length: int=None):
    '''
    dir_list에 포함되는 디렉토리들을 순회하며 그래프를 그리는 함수.
    target_csv_list가 None이 아닌 경우, 해당 리스트에 있는 csv파일만 그래프로 그린다. 

    해당 함수는 각각의 데이터를 플롯하는 함수이다. 즉, 데이터 비교를 위해 하나의 그래프에 여러 데이터를 그리고자 한다면
    다른 함수를 이용해야 한다.
    '''
    for directory in sorted(dir_list):    
        # 여기서 axis에 따라 반복문 다르게 할 수 있음!
        axis = directory[-1]
        csv_file_list = config.axis_to_csv_dic[axis]
        print(csv_file_list)
        
        if target_csv_list:
            filtered_csv_file_list = []

            # csv_file_list에 있는 파일들 중에서 target_csv_list에 포함된 파일만 필터링
            for csv_file in csv_file_list:
                if csv_file in target_csv_list: # csv_file이 target_csv_list에 있으면
                    filtered_csv_file_list.append(csv_file) # 해당 파일을 filtered_csv_file_list에 추가
            csv_file_list = filtered_csv_file_list
        
        if not len(csv_file_list):
            raise ValueError(f'The CSV file that meets your criteria does not exist. No graph will be generated.')

        for csv_file in csv_file_list:

            ## 여기서 get_data를 진행함 -> 그냥 그대로 가져와야 하는데
            data = get_data(directory, csv_file, config.axis_to_marker_dic[axis])
            for column in config.axis_to_marker_dic[axis]:
                draw_data(data=data, csv_file=csv_file, column=column, axis=axis, length=length)

def check_param(target_view: str = None, target_fault_type: str = None):
    '''
    target_view, target_fault_type가 제대로 된 값이 들어왔는지 확인하는 함수
    '''
    valid_views = config.axis_list
    valid_fault_type = config.fault_type_list

    if target_view and target_view not in valid_views:
        raise ValueError(f'target_view must be one of the {valid_views}. You entered {target_view}')
    if target_fault_type and target_fault_type not in valid_fault_type:
        raise ValueError(f'target_fault_type must be one of the {valid_fault_type}. You entered {target_fault_type}')



