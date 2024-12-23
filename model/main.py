# main.py

import os

import tensorflow as tf
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report

from funs.databuilder import make_dataframe, augment_dataframe, add_statistics, get_data_label
from funs.utils import set_seed, load_yaml, get_dir_list, log_results, calculate_result
from funs.draw import get_stat_hist_pic, get_displacement_pic
from funs.Model import ANN
from funs.Trainer import Trainer

import numpy as np

import datetime



def main(
        yaml_config, target_config, save_figs=False, save_model=False, 
        save_log=False, save_displacement_figs=False
        ):
    
    ##############################
    # 1. initialize              #
    ##############################
    set_seed(yaml_config.seed)


    ##############################
    # 2. preprocessing           #
    ##############################
    directory_list = [os.path.join(yaml_config.output_dir, date) for date in target_config['date']]
    directory = get_dir_list(directory_list, target_view=target_config['axis'])

    # 데이터프레임 제작
    df = make_dataframe(yaml_config, directory)

    # 데이터 증강
    augmented_df = augment_dataframe(df, ['z'], yaml_config.sample_size, yaml_config.overlap)

    # 통계값 값 추가
    statistics_df = add_statistics(augmented_df, ['z'])

    print("총 데이터 개수:", len(statistics_df))
    fault_type_counts = statistics_df["fault_type"].value_counts()
    print(f"결함 별 데이터 개수:\n{fault_type_counts}") 
    

    ##############################
    # 3. plot figs (optional)    #
    ##############################
    # 변위 데이터 플롯
    if save_displacement_figs:
        get_displacement_pic(statistics_df, 'z', target_config['date'])

    # 특징별 분포 히스토그램 플롯
    if save_figs:
        date_str = "_".join([str(date) for date in target_config["date"]])
        get_stat_hist_pic(statistics_df, 
                          main_title=f'All feature distribution, z axis, Front view',
                          draw_targets=list(statistics_df.columns),
                          save_path=f'./feature_distribution_figs/z_axis/{date_str}_feature_distribution_peak_rms.png')
    

    ##############################
    # 4. train                   #
    ##############################
    # 데이터, 라벨 얻기
    X, Y = get_data_label(statistics_df, target_config['input_feature'])
    print(f'input feature: {target_config["input_feature"]}')

    model = ANN()
    trainer = Trainer(yaml_config)
    
    accuracies, losses, all_y_true, all_y_pred = trainer.kfold_training(X, Y, model)
    mean_accuracy, accuracy_confidence_interval, mean_loss, loss_confidence_interval = calculate_result(accuracies, losses)

    # 전체 테스트 결과를 기반으로 성능 보고서 출력
    all_y_true = np.concatenate(all_y_true, axis=0)
    all_y_pred = np.concatenate(all_y_pred, axis=0)

    report = classification_report(all_y_true, all_y_pred, target_names=['B', 'H', 'IR', 'OR'], digits=4)
    print('클래스별 성능 보고서')
    print(report)

    class_accuracies = {}
    for class_label in np.unique(all_y_true):
        correct_class_predictions = np.sum((all_y_true == class_label) & (all_y_pred == class_label))
        total_class_samples = np.sum(all_y_true == class_label)
        class_accuracies[class_label] = correct_class_predictions / total_class_samples if total_class_samples > 0 else 0

    print("클래스별 정확도:")
    for class_label, accuracy in class_accuracies.items():
        print(f"클래스 {yaml_config.class2label_dic[class_label]}: {accuracy:.4f}")


    ##############################
    # 5. save                    #
    ##############################
    if save_log:
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_results(
            file_path = yaml_config['log_txt'],
            timestamp=current_time,
            date = target_config['date'],
            input_feature=target_config['input_feature'],
            mean_accuracy=mean_accuracy,
            accuracy_confidence_interval=accuracy_confidence_interval,
            mean_loss=mean_loss,
            loss_confidence_interval=loss_confidence_interval,
            class2label_dic = yaml_config.class2label_dic,
            class_accuracies = class_accuracies,
            report=report,
        )

    # 모델 저장
    if save_model:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()

        model_dir = yaml_config.model_dir
        model_filename = f"{'ANN'}_{target_config['input_feature']}.tflite"
        model_path = os.path.join(model_dir, model_filename)

        os.makedirs(model_dir, exist_ok=True)
        with open(model_path, 'wb') as f:
            f.write(tflite_model)



if __name__ == "__main__":
    target_config = {
        'date': ['1217'],  
        # 'date': ['1011', '1012', '1024', '1102', '1105', '1217'],         # 필수
        # 'bearing_type': '6204', # optional
        # 'RPM': '1200',          # optional
        'axis': 'F',            # optional
        'input_feature': 'z_rms'  # 필수. 모델 input feature로 사용할 데이터
    }

    yaml_config = load_yaml('./model_config.yaml')

    main(yaml_config, target_config, save_figs=True, save_model=False, save_log=False)
