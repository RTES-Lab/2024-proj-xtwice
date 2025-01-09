# draw.py

import pandas as pd
import matplotlib.pyplot as plt

from typing import List

def get_stat_hist_pic(df: pd.DataFrame, main_title: str, draw_targets: List[str], save_path: str):
    """
    결함 별 특징 히스토그램을 그리는 함수
    """
    fault_type_list = ['H', 'OR', 'B', 'IR']

    # draw_targets를 rms와 peak로 분리
    rms_targets = [target for target in draw_targets if 'rms' in target.lower()]
    skewness_targets = [target for target in draw_targets if 'skewness' in target.lower()]
    kurtosis_targets = [target for target in draw_targets if 'kurtosis' in target.lower()]

    groups = {"RMS": rms_targets, "Skewness": skewness_targets, "Kurtosis": kurtosis_targets}
    
    fig_len = len(groups)
    plt.figure(figsize=(20, 5 * fig_len))

    plt.suptitle(main_title, fontsize=50, y=0.99)

    for i, (group_name, targets) in enumerate(groups.items()):
        plt.subplot(fig_len, 1, i + 1)
        for fault_type in fault_type_list:
            # if fault_type == 'B':
            #     continue
            for target in targets:
                subset = df[df['fault_type'] == fault_type]
                plt.hist(subset[target], bins=50, alpha=0.5, label=f"{fault_type} ({target})")
        plt.title(f'{group_name} Distribution', size=40)
        plt.xlabel(f'{group_name} Values', size=30)
        plt.ylabel('Counts', size=30)
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)
        plt.legend(fontsize=15, loc='upper right')  # 레이블을 그래프 오른쪽 외부로 이동
    
    plt.tight_layout()  
    plt.savefig(save_path)
    plt.close()
    print(f"그림이 {save_path}에 저장되었습니다.")


def get_displacement_pic(df: pd.DataFrame, axis: str, date: str):
    """
    결함 별 변위 데이터를 그리는 함수
    """
    grouped = df.groupby('fault_type')

    for fault_type, group in grouped:
        # 각 fault_type의 z 값 결합
        displacement_values = [item for sublist in group[axis] for item in sublist]

        # 플롯 생성
        plt.figure(figsize=(15, 6))
        plt.plot(displacement_values, label=f"{fault_type}", linewidth=2)
        plt.title(f"{axis} Values for Fault Type: {fault_type}", fontsize=16)
        plt.xlabel("Index", fontsize=12)
        plt.ylabel(f"{axis} Value", fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True)

        # 파일 이름 설정 및 저장
        file_name = f"{int(date)}_{fault_type}_{axis}_plot.png"
        plt.savefig(file_name, bbox_inches='tight')
        plt.close()

        print(f"{fault_type}의 변위 그림이 {file_name}에 저장되었습니다.")