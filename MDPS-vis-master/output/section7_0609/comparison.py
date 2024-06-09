"""
x.csv, y.csv 파일로부터 그래프를 그리는 함수.
selected_folders에 입력된 폴더의 csv 파일 데이터를 한 그래프로 그려
쉬운 비교를 가능하게 한다.

이 파일은 비교하고자 하는 섹션 이름 폴더 안에 위치해야 한다.
예를 들어, 1번 섹션에서 30204 B과 30204 H의 변위를 그래프로 그리고 싶다면
comparison.py는 section1 디렉토리 안에 위치해야만 한다.
"""

import pandas as pd
import matplotlib.pyplot as plt
import os

from typing import Tuple, List

def get_data(
        root_dir: str, folder_list:list, csv_file: str, columns: list
        ) -> List[Tuple[str, pd.DataFrame]]:
    """
    데이터를 얻어오는 함수
    csv 파일을 읽은 다음 각 열의 평균을 빼는 데이터 중심화를 진행한 뒤 해당 값을 data에 추가한다.
    data는 folder_list 안에 있는 csv 파일들의 경로화 데이터 값을 갖는 리스트이다.

    Parameters
    ----------
    root_dir: str
        상위 폴더 경로
    folder_list: list
        비교하고자 하는 데이터가 있는 폴더 이름 리스트
    csv_file: str
        csv 파일 이름 
    columns: list
        csv 파일의 column 리스트

    Returns
    ----------
    data: List[Tuple[str, pd.DataFrame]]
        (파일 경로, 해당 파일의 데이터프레임) 튜플을 원소값으로 하는 리스트

    Examples
    ----------
    >>> script_path = os.path.dirname(os.path.realpath(__file__))
    >>> root_dir = os.path.dirname(script_path)
    >>> selected_folders = ['0530_30204_OR_1200RPM_120fps_1','0530_30204_H_1200RPM_120fps_1']
    >>> csv_file = 'x.csv'
    >>> columns = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    >>> data = get_data(root_dir=root_dir, folder_list=selected_folders, csv_file=csv_file, columns=columns)
    """

    root_dir = root_dir

    selected_folders = folder_list

    data = []

    for subdir in selected_folders:
        folder_path = os.path.join(root_dir, subdir)
        file_path = os.path.join(folder_path, csv_file)

        df = pd.read_csv(file_path, usecols=list(range(0, len(columns))), names=columns, header=0) 

        means = df.mean()

        df_centered = df - means

        data.append((file_path, df_centered))
            
    return data

def compare_data(data: list, csv_file: str, column: str, bearing_type: str, section_type: str):
    """
    데이터를 하나의 그래프에 그리는 함수

    Parameters
    ----------
    data: list
        (파일 경로, 해당 파일의 데이터프레임) 튜플을 원소값으로 하는 리스트
    csv_file: str
        csv 파일 이름 
    column: str
        csv 파일의 column
    bearing_type: str
        비교하고자 하는 두 개의 베어링 타입
    section_type: str
        비교하고자 하는 섹션 이름

    Examples
    ----------
    >>> data = get_data(root_dir=root_dir, folder_list=selected_folders, csv_file=csv_file, columns=columns)
    >>> csv_file = 'x.csv'
    >>> column = 'A'
    >>> bearing_type = bearing_type = 'OR_and_H'
    >>> section_type ='section1'
    >>> compare_data(data=data, csv_file=csv_file, column=column, bearing_type=bearing_type, section_type=section_type)
    """
    plt.figure(figsize=(15,6))

    # 비교하고자 하는 두 데이터의 측정 시간이 다를 경우, 시간이 짧은 쪽에 맞춤
    min_len = -1
    for _, df in data:
        if min_len > len(df) or min_len < 0:
            min_len = len(df) 

    for subdir, df in data:
        df = df[:min_len]
        bearing = next((x for x in ['OR', 'H', 'IR', 'B'] if x in subdir), subdir)
        if bearing == 'OR':
            mm_per_pixel = 0.111
        if bearing == 'H':
            mm_per_pixel = 0.098
        if bearing == 'IR':
            mm_per_pixel = 0.075
        if bearing == 'B':
            mm_per_pixel = 0.103
        plt.plot(df.index / 120, df[column] * mm_per_pixel, label=bearing)

    plt.title(f'Comparison of {column}, {bearing_type.replace("_", " ")}', size=15)
    plt.xlabel('Time[s]', size=15)
    plt.ylabel('Displacement[mm]', size=15)
    plt.ylim(-0.1, 0.1)
    plt.legend()    
    plt.savefig(os.path.join(script_path, f'comparison_{column}_{bearing_type}_{csv_file[0].upper()}_axis_{section_type}.png'))  # 그래프 저장
    plt.close()  
    print("Graphs saved successfully.")

def get_compared_data(csv_files, columns, bearing_type, section_type):
    """
    두 개의 폴더에 있는 데이터를 한꺼번에 그리는 함수

    Parameters
    ----------
    csv_files: list
        (파일 경로, 해당 파일의 데이터프레임) 튜플을 원소값으로 하는 리스트
    columns: str
        csv 파일 이름 리스트
    column: str
        csv 파일의 column 리스트
    bearing_type: str
        비교하고자 하는 두 개의 베어링 타입
    section_type: str
        비교하고자 하는 섹션 이름

    Examples
    ----------
    >>> selected_folders = ['0530_30204_OR_1200RPM_120fps_1','0530_30204_H_1200RPM_120fps_1']
    >>> bearing_type = 'OR_and_H'
    >>> section_type = 'section1'
    >>> columns = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    >>> csv_files = ['x.csv', 'y.csv']
    >>> get_compared_data(csv_files, columns, bearing_type, section_type)
    """
    for csv_file in csv_files:
        data = get_data(root_dir=root_dir, folder_list=selected_folders, csv_file=csv_file, columns=columns)
        for column in columns:
            compare_data(data=data, csv_file=csv_file, column=column, bearing_type=bearing_type, section_type=section_type)


if __name__=="__main__":

    # 현재 스크립트의 경로를 가져옴
    script_path = os.path.dirname(os.path.realpath(__file__))

    # 해당 경로로 이동
    os.chdir(script_path)

    # 상위 폴더 경로
    root_dir = os.path.dirname(script_path)

    # selected_folders = ['0609_30204_B_1200RPM_120fps_7','0609_30204_H_1200RPM_120fps_7']
    # bearing_type = 'B_and_H'

    # selected_folders = ['0609_30204_IR_1200RPM_120fps_7','0609_30204_H_1200RPM_120fps_7']
    # bearing_type = 'IR_and_H'

    # selected_folders = ['0609_30204_OR_1200RPM_120fps_7','0609_30204_H_1200RPM_120fps_7']
    # bearing_type = 'OR_and_H'


    selected_folders = ['0609_30204_IR_1200RPM_120fps_7','0609_30204_OR_1200RPM_120fps_7']
    bearing_type = 'IR_and_OR'

    section_type = 'section7'

    if selected_folders[0][-1] == "1":
        columns = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    else:
        columns = ['A', 'B']

    if selected_folders[0][-1] in ["4", "5", "7"]:
        csv_files = ['y.csv', 'z.csv']
    else:
        csv_files = ['x.csv', 'y.csv']

    get_compared_data(csv_files, columns, bearing_type, section_type)
