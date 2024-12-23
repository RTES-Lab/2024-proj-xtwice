# draw.py

import pandas as pd
import matplotlib.pyplot as plt

from typing import List

def get_stat_hist_pic(df: pd.DataFrame, main_title: str, draw_targets: List[str], save_path: str):
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
    import matplotlib.pyplot as plt

    fault_type_list = ['H', 'OR', 'B', 'IR']

    # draw_targets를 rms와 peak로 분리
    rms_targets = [target for target in draw_targets if 'rms' in target.lower()]
    peak_targets = [target for target in draw_targets if 'peak' in target.lower()]

    groups = {"RMS": rms_targets, "Peak": peak_targets}
    
    fig_len = len(groups)
    plt.figure(figsize=(20, 5 * fig_len))

    plt.suptitle(main_title, fontsize=50)

    for i, (group_name, targets) in enumerate(groups.items()):
        plt.subplot(fig_len, 1, i + 1)
        for fault_type in fault_type_list:
            for target in targets:
                subset = df[df['fault_type'] == fault_type]
                plt.hist(subset[target], bins=50, alpha=0.5, label=f"{fault_type} ({target})")
        plt.title(f'{group_name} Distribution', size=40)
        plt.xlabel(f'{group_name} Values', size=30)
        plt.ylabel('Counts', size=30)
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)
        plt.legend(fontsize=15, loc='upper right')  # 레이블을 그래프 오른쪽 외부로 이동

    # 레이아웃 조정 및 저장
    # plt.tight_layout(rect=[0, 0, 0.85, 0.95])  # 그래프 오른쪽에 공간 확보
    # plt.savefig(save_path, bbox_inches='tight')
    
    plt.tight_layout()  # 그래프 오른쪽에 공간 확보
    plt.savefig(save_path)
    plt.close()
    print(f"그림이 {save_path}에 저장되었습니다.")