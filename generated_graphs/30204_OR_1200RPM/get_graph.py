import pandas as pd
import matplotlib.pyplot as plt
import os


filename = os.getcwd().split('\\')[4]


def get_graph(csv_file : str):

    file = csv_file  # 파일 경로를 지정하세요.
    df = pd.read_csv(file, usecols=[0, 1, 2, 3], names=['A', 'B', 'C', 'D'], header=0)  # 첫 4개 열만 불러옵니다.
    df['B'] =  df['B'][185:]

    # 각 열의 평균을 구합니다.
    means = df.mean()


    # 각 열의 모든 값에서 각 열의 평균을 뺍니다.
    df_centered = df - means

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
