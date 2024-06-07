"""
x.csv, y.csv 파일로부터 그래프를 그리는 함수.
이 파일은 그래프를 그리고자 하는 베어링의 폴더 안에 위치해야 한다.
예를 들어, 30204 B 베어링의 1번 위치에서 추출한 x, y 방향 변위 그래프를 확인하고 싶다면 
get_graph.py는 0530_30204_B_1200RPM_120fps_1 디렉토리 안에 위치해야만 한다.
"""

import pandas as pd
import matplotlib.pyplot as plt
import os

# 현재 스크립트의 경로를 가져옴
script_path = os.path.dirname(os.path.realpath(__file__))

# 해당 경로로 이동
os.chdir(script_path)

# 파일 이름 정의
filename = os.getcwd().split('/')[-1]

def get_graph(csv_file : str):
    """
    그래프를 그리는 함수. 
    csv 파일을 읽은 다음 각 열의 평균을 빼는 데이터 중심화를 진행한 뒤 해당 값으로 그래프를 그린다.
    csv 파일을 읽을 때 읽어들일 데이터와(usecols) 각각의 이름(names)를 정의해야 한다.
    가령, 4개 스티커의 변위를 추출했다면 usecols=[0, 1, 2, 3], names=['A', 'B', 'C', 'D']이 되어야 한다.

    Parameters
    ----------
    csv_file: str
        사용할 csv파일. y.csv 혹은 x.csv

    Examples
    ----------
    >>> $ cd ./MDPS-vis-master/output/0530_30204_B_1200RPM_120fps_1
    >>> $ python get_graph.py
    """

    file = csv_file 
    df = pd.read_csv(file, usecols=[0, 1, 2, 3], names=['A', 'B', 'C', 'D'], header=0)  # 네 개의 데이터를 읽는 경우
                                                                                        # 즉, 스티커 네 개에서 추출된 변위를 사용하는 경우
    # 각 열의 평균 계산
    means = df.mean()

    # 데이터 중심화
    df_centered = df - means

    # 그래프 생성
    for column in df_centered.columns:
        plt.figure(figsize=(15, 6))
        plt.plot(df_centered.index/120, df_centered[column]*0.0722222)
        plt.title(f'Centered Values of {column}', size=15)
        plt.xlabel('Time[s]', size=15)
        plt.ylabel('Displacement[mm]', size=15)
        plt.savefig(f'{filename}_{column}_{csv_file[0].upper()}_axis.png')
        plt.close()

    print("Graphs saved successfully.")


get_graph('y.csv')
get_graph('z.csv')
