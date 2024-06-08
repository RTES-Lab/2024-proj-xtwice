import pandas as pd
import matplotlib.pyplot as plt
import os

def get_data(root_dir, folder_list:list, csv_file):

    # 루트 디렉토리 경로
    root_dir = root_dir

    # 선택할 폴더 이름들
    selected_folders = folder_list

    # 데이터를 저장할 리스트 초기화
    data = []

    # 선택한 폴더들만 순회합니다.
    for subdir in selected_folders:
        folder_path = os.path.join(root_dir, subdir)
        file_path = os.path.join(folder_path, csv_file)
        
        # CSV 파일을 불러옵니다.
        df = pd.read_csv(file_path, usecols=[0,1,2,3], names=['A','B','C','D'], header=0)[:765]  # A열만 불러옵니다.

            # 각 열의 평균을 구합니다.
        means = df.mean()

        # 각 열의 모든 값에서 각 열의 평균을 뺍니다.
        df_centered = df - means


        # 데이터 리스트에 추가
        data.append((subdir, df_centered))
        
    return data

def compare_data(data, column : str, csv_file):
    plt.figure(figsize=(15,6))
    for subdir, df in data:
        if 'OR' in subdir:
            subdir = 'OR'
        if 'H' in subdir:
            subdir = 'H'
        if 'IR' in subdir:
            subdir = 'IR'
        if 'B' in subdir:
            subdir = 'B'
        plt.plot(df.index/120, df[column]*0.0722222, label=subdir)

    # 그래프 설정
    plt.title(f'Comparison of {column}', size=15)
    plt.xlabel('Time[s]', size=15)
    plt.ylabel('Displacement[mm]', size=15)
    plt.ylim(-0.1, 0.1)
    plt.legend()  # 범례 추가
    plt.savefig(os.path.join(root_dir, f'comparison_{column}_H_and_OR_{csv_file[0].upper()}_axis.png'))  # 그래프 저장
    print("Graphs saved successfully.")

def get_compared_data(csv_file):
    data = get_data(root_dir=root_dir, folder_list=selected_folders, csv_file= csv_file)

    compare_data(data,'A', csv_file)
    compare_data(data,'B', csv_file)
    compare_data(data,'C', csv_file)
    compare_data(data,'D', csv_file)


root_dir = 'C:\Github\X-twice\generated_graphs\\'
selected_folders = ['30204_OR_1200RPM', '30204_H_1200RPM']
get_compared_data('y.csv')
get_compared_data('z.csv')

