# utils.py

import random
import os
import glob

import yaml
from box import Box

from typing import List, Optional

import torch
import tensorflow as tf
import numpy as np
import scipy.stats as stats

def load_yaml(config_path: str) -> Box:
    """
    YAML 파일을 load하는 함수

    Parameters
    ----------
    config_path : str
        YAML 파일 경로

    Returns
    -------
    Box 
        Box 개체
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config_yaml = yaml.load(f, Loader=yaml.FullLoader)
        config = Box(config_yaml)

    return config


def set_seed(seed: int):
    """
    랜덤 시드 고정 함수.

    Parameters
    ----------
    seed : int
        고정할 시드 값
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    tf.random.set_seed(seed) 
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


from typing import List, Optional, Union
import os
import glob

def get_dir_list(
        target_dir: Union[str, List[str]], 
        target_bearing: Optional[str] = None, target_rpm: Optional[str] = None, 
        target_fault_type: Optional[str] = None, target_view: Optional[str] = None, 
        ) -> List[str]:
    '''
    주어진 조건에 맞는 디렉토리 리스트를 반환하는 함수
    
    Parameters
    ----------
    target_dir: Union[str, List[str]]
        검색할 상위 디렉토리 경로. 문자열 또는 문자열 리스트 가능.
    target_bearing: str
        디렉토리 이름에 포함될 베어링 유형 정보. 기본값은 None.
    target_rpm: str, optional
        디렉토리 이름에 포함될 rpm 유형 정보. 기본값은 None.
    target_fault_type: str, optional
        디렉토리 이름에 포함될 결함 유형 정보. 기본값은 None.
    target_view: str, optional
        디렉토리 이름에 포함될 뷰 정보. 기본값은 None.

    Returns
    ----------
    dir_list: List[str]  
        조건에 맞는 디렉토리 경로 리스트.
    '''
    # target_dir이 문자열이면 리스트로 변환
    if isinstance(target_dir, str):
        target_dir = [target_dir]

    # 모든 target_dir에 대해 디렉토리 패턴 검색
    all_dir_list = []
    for base_dir in target_dir:
        pattern = os.path.join(base_dir, '*')

        if target_bearing:
            pattern += f'{target_bearing}*'
        if target_rpm:
            pattern += f'{target_rpm}*'
        if target_fault_type:
            pattern += f'{target_fault_type}*'
        if target_view:
            pattern += f'{target_view}'

        dir_list = glob.glob(pattern)
        all_dir_list.extend(dir_list)

    if not all_dir_list:
        raise FileNotFoundError('The directories that meet your criteria do not exist. Please check the parameters you entered.')

    return sorted(all_dir_list)

def log_results(file_path, input_feature, mean_accuracy, accuracy_confidence_interval, mean_loss, loss_confidence_interval, report):
    """
    결과를 파일에 저장하거나 기존 파일에 이어 쓰는 함수.

    Parameters
    ----------
    file_path : str
        저장할 파일 경로
    input_feature : str
        모델에서 사용한 특징 (target_config['input_feature'])
    mean_accuracy : float
        평균 정확도
    accuracy_confidence_interval : Tuple[float, float]
        정확도 신뢰구간
    mean_loss : float
        평균 손실
    loss_confidence_interval : Tuple[float, float]
        손실 신뢰구간
    report : str
        성능 보고서 (classification_report 출력)
    """
    with open(file_path, 'a') as file:  # 'a' 모드로 열어 기존 파일에 이어 쓰기
        file.write("====================================================\n")
        file.write("모델 결과\n")
        file.write(f"사용 특징: {input_feature}\n")
        file.write(f"정확도: {mean_accuracy:.4f} ± {accuracy_confidence_interval[1] - mean_accuracy:.4f}\n")
        file.write(f"손실: {mean_loss:.4f} ± {loss_confidence_interval[1] - mean_loss:.4f}\n")
        file.write("\n클래스별 성능 보고서:\n")
        file.write(report)
        file.write("\n")
    print(f"결과가 {file_path}에 저장되었습니다.")


def calculate_result(accuracy, loss):
    if len(accuracy) > 1:
        mean_accuracy = np.mean(accuracy)
        accuracy_variance = np.var(accuracy)
        mean_loss = np.mean(loss)
        loss_variance = np.var(loss)

        # 신뢰구간 계산
        if accuracy_variance > 0:
            accuracy_confidence_interval = stats.t.interval(0.95, len(accuracy)-1, loc=mean_accuracy, scale=stats.sem(accuracy))
            print(f"정확도: {mean_accuracy:.4f} ± {accuracy_confidence_interval[1] - mean_accuracy:.4f}")
        else:
            print(f"정확도: {mean_accuracy:.4f} (변동이 없어 신뢰구간을 계산할 수 없습니다.)")

        if loss_variance > 0:
            loss_confidence_interval = stats.t.interval(0.95, len(loss)-1, loc=mean_loss, scale=stats.sem(loss))
            print(f"손실: {mean_loss:.4f} ± {loss_confidence_interval[1] - mean_loss:.4f}")
        else:
            print(f"손실: {mean_loss:.4f} (변동이 없어 신뢰구간을 계산할 수 없습니다.)")

    return mean_accuracy, accuracy_confidence_interval, mean_loss, loss_confidence_interval
