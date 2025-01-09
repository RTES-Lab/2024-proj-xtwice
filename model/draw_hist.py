# main.py

import os

import funs

import numpy as np
import pandas as pd

import argparse


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
    """
    각 데이터프레임에 통계 특성을 추가하고, train_df의 각 특성별 평균과 표준편차를 기준으로 표준화합니다.
    
    Args:
        train_df: 학습 데이터프레임
        val_df: 검증 데이터프레임
        test_df: 테스트 데이터프레임
        target_axis: 통계를 계산할 축 목록
        is_standardize: 표준화 여부
    
    Returns:
        처리된 train_df, val_df, test_df를 튜플로 반환
    """
    
    # train 데이터의 특성별 평균과 표준편차를 저장할 딕셔너리
    train_stats = {}
    
    for axis in target_axis:
        if axis not in df.columns:
            raise ValueError(f"Target axis '{axis}' not found in DataFrame columns.")

        # 각 축에 대해 결과 리스트 초기화
        rms_values = []
        peak_values = []
        skewness_values = []
        kurtosis_values = []
        fused_feature_values = []
        
        # 통계 값 계산
        for sample in df[axis]:
            peak, _, rms, _, _, skew, kurt = calculate_statistics(sample)
            rms_values.append(rms)
            peak_values.append(peak)
            skewness_values.append(skew)
            kurtosis_values.append(kurt)
            fused_feature_values.append([rms, skew, kurt])
        
        if is_standardize:
            train_stats[axis] = {
                'rms': {'mean': np.mean(rms_values), 'std': np.std(rms_values)},
                'skewness': {'mean': np.mean(skewness_values), 'std': np.std(skewness_values)},
                'kurtosis': {'mean': np.mean(kurtosis_values), 'std': np.std(kurtosis_values)},
                'peak': {'mean': np.mean(peak_values), 'std': np.std(peak_values)}
            }
            
            # train에서 계산된 평균과 표준편차로 각 특성별 표준화
            stats = train_stats[axis]
            
            # RMS 표준화
            standardized_rms = (np.array(rms_values) - stats['rms']['mean']) / stats['rms']['std']
            
            # Skewness 표준화
            standardized_skew = (np.array(skewness_values) - stats['skewness']['mean']) / stats['skewness']['std']
            
            # Kurtosis 표준화
            standardized_kurt = (np.array(kurtosis_values) - stats['kurtosis']['mean']) / stats['kurtosis']['std']
            
            # Peak 표준화
            standardized_peak = (np.array(peak_values) - stats['peak']['mean']) / stats['peak']['std']
            
            # 표준화된 값 데이터프레임에 추가
            df[f'{axis}_rms'] = standardized_rms
            df[f'{axis}_skewness'] = standardized_skew
            df[f'{axis}_kurtosis'] = standardized_kurt
            df[f'{axis}_peak'] = standardized_peak

        
            # fused_feature값도 표준화된 값으로 갱신
            for i in range(len(df)):
                fused_feature_values[i] = [
                    standardized_rms[i],
                    standardized_skew[i],
                    standardized_kurt[i]
                ]
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

    # 데이터프레임 제작
    df = {
        "1105": [],
        "1217": [],
        "0108": []
    }
    valid_dfs = []

    date_list = target_config['date']

    for date in date_list:
        directory_list = [os.path.join(yaml_config.output_dir, date)]
        directory = funs.get_dir_list(directory_list, target_view=target_config['view'])
        df[date] = funs.make_dataframe(yaml_config, directory)
        df[date] = funs.augment_dataframe(df[date], target_config['axis'], yaml_config.sample_size, yaml_config.overlap)
        df[date] = add_statistics(df[date], target_config['axis'], is_standardize=True)
        
        if isinstance(df[date], pd.DataFrame) and not df[date].empty:
            valid_dfs.append(df[date])

    # 유효한 데이터프레임 합치기
    if valid_dfs:
        df_combined = pd.concat(valid_dfs, ignore_index=True)
    else:
        df_combined = pd.DataFrame()  # 빈 데이터프레임


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
    funs.get_stat_hist_pic(df_combined, 
                        main_title=f'{date_name} feature distribution, {axis_name} axis, Front view',
                        draw_targets=list(df_combined.columns),
                        save_path=f'{yaml_config.feature_figs_dir}/{axis_name}_axis/standardization/{date_str}_feature_distribution_peak_rms.png')
        
def draw_train_hist(yaml_config, target_config):
    
    directory_list = [os.path.join(yaml_config.output_dir, date) for date in target_config['date']]
    directory = funs.get_dir_list(directory_list, target_view=target_config['view'])

    # 데이터프레임 제작
    df = funs.make_dataframe(yaml_config, directory)

    train_df, val_df, test_df = funs.split_dataframe(df, 0.7, 0.3)

    # 데이터 증강
    train_df = funs.augment_dataframe(train_df, target_config['axis'], yaml_config.sample_size, yaml_config.overlap)
    val_df = funs.augment_dataframe(val_df, target_config['axis'], yaml_config.sample_size, yaml_config.overlap)
    test_df = funs.augment_dataframe(test_df, target_config['axis'], yaml_config.sample_size, yaml_config.overlap)

    # 통계값 값 추가
    train_df, val_df, test_df = funs.add_statistics(train_df, val_df, test_df, target_config['axis'], is_standardize=True)
    df_combined = pd.concat([train_df, val_df, test_df], ignore_index=True)

    print("총 데이터 개수:", len(train_df)+len(val_df)+len(test_df))
    fault_type_counts = train_df["fault_type"].value_counts()
    fault_type_counts += val_df["fault_type"].value_counts()
    fault_type_counts += test_df["fault_type"].value_counts()
    print(f"결함 별 데이터 개수:\n{fault_type_counts}") 

    funs.get_stat_hist_pic(train_df, 
                        main_title=f'train feature distribution',
                        draw_targets=list(train_df.columns),
                        save_path=f'{yaml_config.feature_figs_dir}/train/train_feature_distribution_peak_rms.png'
                        )
    funs.get_stat_hist_pic(val_df, 
                    main_title=f'validation feature distribution',
                    draw_targets=list(val_df.columns),
                    save_path=f'{yaml_config.feature_figs_dir}/train/val_feature_distribution_peak_rms.png'
                    )
    funs.get_stat_hist_pic(test_df, 
                    main_title=f'test feature distribution',
                    draw_targets=list(test_df.columns),
                    save_path=f'{yaml_config.feature_figs_dir}/train/test_feature_distribution_peak_rms.png'
                    )
    funs.get_stat_hist_pic(df_combined, 
                    main_title=f'All feature distribution',
                    draw_targets=list(df_combined.columns),
                    save_path=f'{yaml_config.feature_figs_dir}/train/all_feature_distribution_peak_rms.png'
                    )
    


def main():
    funs.set_seed(yaml_config.seed)

    # 실행할 함수 선택
    if args.mode == "original":
        draw_original_hist(yaml_config, target_config)
    elif args.mode == "train":
        draw_train_hist(yaml_config, target_config)


    

if __name__ == "__main__":
    args = parse_arguments()

    target_config = {
        'date': args.dates,
        'view': args.view,
        'axis': args.axis,
    }

    yaml_config = funs.load_yaml('./model_config.yaml')

    main()
