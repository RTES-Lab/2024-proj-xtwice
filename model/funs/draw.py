# draw.py

import pandas as pd
import matplotlib.pyplot as plt

from typing import List

def get_stat_hist_pic(df: pd.DataFrame, draw_targets: List[str], save_path: str):
    """
    주어진 데이터프레임에서 각 fault_type에 대한 Peak과 RMS 값의 분포를 히스토그램으로 생성하고 저장하는 함수

    Parameters
    ----------
    df : pd.DataFrame
        통계값이 포함된 데이터프레임
    draw_targets: List[str]
        데이터프레임에서 분포를 시각화하고자 하는 열 리스트
    save_path : str
        히스토그램 이미지를 저장할 파일 경로
    """
    fault_type_list = ['H', 'OR', 'B', 'IR']
    
    fig_len = len(draw_targets)
    plt.figure(figsize=(20, 5*fig_len))
    
    for i in range(fig_len):
        plt.subplot(fig_len, 1, i+1)
        for fault_type in fault_type_list:
            subset = df[df['fault_type'] == fault_type]
            plt.hist(subset[draw_targets[i]], bins=50, alpha=0.5, label=fault_type)
        plt.title(f'{draw_targets[i]} Distribution', size=40)  
        plt.xlabel(f'{draw_targets[i]} [mm]', size=30)  
        plt.ylabel('Counts', size=30)  
        plt.xticks(fontsize=25)  
        plt.yticks(fontsize=25)  
        plt.legend(fontsize=25, loc='upper right') 

    # 레이아웃 조정 및 저장
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"히스토그램이 {save_path}에 저장되었습니다.")