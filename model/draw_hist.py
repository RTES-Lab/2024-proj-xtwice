# main.py

import os

import funs

import numpy as np
import pandas as pd

import argparse

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer

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


def parse_arguments():
    parser = argparse.ArgumentParser(description="Draw.")
    parser.add_argument('--dates', nargs='+', required=True, help="Target dates (e.g., 1105 1217)")
    parser.add_argument('--view', type=str, default='F', help="View type (e.g., F)")
    parser.add_argument('--axis', nargs='+', default=['z'], help="Target axis (e.g., z or z x)")
    parser.add_argument("--mode", choices=["original", "train"], required=True, help="Mode to execute")
    return parser.parse_args()

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

def add_statistics(
    df,
    target_axis,
    is_standardize: bool = True
):
    
    # train 데이터의 특성별 평균과 표준편차를 저장할 딕셔너리
    
    for axis in target_axis:
        if axis not in df.columns:
            raise ValueError(f"Target axis '{axis}' not found in DataFrame columns.")
        # 각 축에 대해 결과 리스트 초기화
        rms_values = []
        peak_values = []
        skewness_values = []
        kurtosis_values = []
        fused_feature_values = []
        
        idx = 0
        idx_list = []
        # 통계 값 계산
        for sample in df[axis]:
            idx += 1
            peak, _, rms, _, _, skew, kurt = calculate_statistics(sample)
            if skew < -12 or kurt > 12:
                idx_list.append(idx)
                continue
            rms_values.append(rms)
            peak_values.append(peak)
            skewness_values.append(skew)
            kurtosis_values.append(kurt)
            fused_feature_values.append([rms, skew, kurt])

        df.drop(idx_list, axis=0, inplace=True)
        
        temp_df = pd.DataFrame({
            f'{axis}_rms': rms_values,
            f'{axis}_skewness': skewness_values,
            f'{axis}_kurtosis': kurtosis_values,
            f'{axis}_peak': peak_values
        })

        if is_standardize:
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
        else:
            # 정규화 없이 원본 값을 추가
            df[f'{axis}_rms'] = rms_values
            df[f'{axis}_skewness'] = skewness_values
            df[f'{axis}_kurtosis'] = kurtosis_values
            df[f'{axis}_peak'] = peak_values

        df[f'{axis}_fused_features'] = fused_feature_values
    
    return df


def draw_original_hist(yaml_config, target_config):
    
    ##############################
    # 1. initialize              
    ##############################
    funs.set_seed(yaml_config.seed)

    directory_list = [os.path.join(yaml_config.output_dir, date) for date in target_config['date']]
    directory = funs.get_dir_list(directory_list, target_view=target_config['view'])

    # 데이터프레임 제작
    df = funs.make_dataframe(yaml_config, directory)


    # 데이터 증강
    df = funs.augment_dataframe(df, target_config['axis'], yaml_config.sample_size, yaml_config.overlap)

    # 통계값 값 추가
    df = add_statistics(df, target_config['axis'], is_standardize=False)

    if len(target_config['axis']) == 1:
        axis_name = target_config['axis'][0]
    elif len(target_config['axis']) == 2:
        axis_name = f"{target_config['axis'][0]}, {target_config['axis'][1]}"

    # 날짜 이름 결정(그림 제목용)
    if len(target_config['date']) == 1:
        date_name = target_config['date'][0]
    else:
        date_name = 'All'
        
    # 특징별 분포 히스토그램 플롯
    date_str = "_".join([str(date) for date in target_config["date"]])
    funs.get_stat_hist_pic(df, 
                        main_title=f'{date_name} feature distribution, {axis_name} axis, Front view',
                        draw_targets=list(df.columns),
                        save_path=f'{yaml_config.feature_figs_dir}/{axis_name}_axis/standardization/{date_str}_feature_distribution_peak_rms.png')
        
def main():
    funs.set_seed(yaml_config.seed)

    # 실행할 함수 선택
    if args.mode == "original":
        draw_original_hist(yaml_config, target_config)



if __name__ == "__main__":
    args = parse_arguments()

    target_config = {
        'date': args.dates,
        'view': args.view,
        'axis': args.axis,
    }

    yaml_config = funs.load_yaml('./model_config.yaml')

    main()
