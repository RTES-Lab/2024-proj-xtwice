# compare_tflite.py

import os

import tensorflow as tf
from sklearn.metrics import classification_report

from funs.databuilder import make_dataframe, augment_dataframe, add_statistics, get_data_label
from funs.utils import set_seed, load_yaml, get_dir_list, log_results, calculate_result
from funs.draw import get_stat_hist_pic
from funs.Model import Model

from sklearn.model_selection import train_test_split

import numpy as np

def main(yaml_config, target_config, save_figs=True, save_model=True, save_log=True):
    set_seed(yaml_config.seed)

    directory_list = [os.path.join(yaml_config.output_dir, date) for date in target_config['date']]
    directory = get_dir_list(directory_list, target_view=target_config['axis'])

    # 데이터프레임 제작
    df = make_dataframe(yaml_config, directory)

    # 데이터 증강
    augmented_df = augment_dataframe(df, 'z', yaml_config.sample_size, yaml_config.overlap)

    # 통계값 값 추가
    statistics_df = add_statistics(augmented_df, 'z')
    
    feature_list = list(statistics_df.columns)[2:]
    feature_list.remove('average')

    if save_figs:
        date_str = "_".join([str(date) for date in target_config["date"]])
        get_stat_hist_pic(statistics_df, 
                          main_title=f'{date_str} feature distribution, z axis, Front view',
                          draw_targets=feature_list,
                          save_path=f'./feature_distribution_figs/{date_str}_feature_distribution_all.png')
        
    # 데이터, 라벨 얻기
    X, Y = get_data_label(statistics_df, target_config['input_feature'])
    print(f'input feature: {target_config["input_feature"]}')

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    accuracies = []
    losses = []
    all_y_true = []
    all_y_pred = []

    # 모델 정의
    model = Model(X_train)
    model, model_name = model.ANN()

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # 모델 학습
    history = model.fit(X_train, y_train, epochs=yaml_config.epochs, batch_size=yaml_config.batch_size, validation_data=(X_test, y_test), verbose=0)
    

    # 검증 정확도 및 손실
    accuracy = history.history['val_accuracy'][-1]
    loss = history.history['val_loss'][-1]

    # 예측값 및 실제값 저장
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    all_y_true.append(y_test)
    all_y_pred.append(y_pred)

    if not np.isnan(accuracy) and not np.isnan(loss):
        accuracies.append(accuracy)
        losses.append(loss)
    
    mean_accuracy, accuracy_confidence_interval, mean_loss, loss_confidence_interval = calculate_result(accuracies, losses)

    # 전체 테스트 결과를 기반으로 성능 보고서 출력
    all_y_true = np.concatenate(all_y_true, axis=0)
    all_y_pred = np.concatenate(all_y_pred, axis=0)

    report = classification_report(all_y_true, all_y_pred, target_names=['B', 'H', 'IR', 'OR'], digits=4)
    print('클래스별 성능 보고서')
    print(report)

    if save_log:
        log_results(
            yaml_config['log_txt'],
            input_feature=target_config['input_feature'],
            mean_accuracy=mean_accuracy,
            accuracy_confidence_interval=accuracy_confidence_interval,
            mean_loss=mean_loss,
            loss_confidence_interval=loss_confidence_interval,
            report=report
        )

    # 모델 저장
    if save_model:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()

        model_dir = yaml_config.model_dir
        model_filename = f"{model_name}_{target_config['input_feature']}.tflite"
        model_path = os.path.join(model_dir, model_filename)

        os.makedirs(model_dir, exist_ok=True)
        with open(model_path, 'wb') as f:
            f.write(tflite_model)


if __name__ == "__main__":
    target_config = {
        'date': ['1105'],  
        # 'date': ['1011', '1012', '1024', '1102', '1105'],         # 필수
        # 'bearing_type': '6204', # optional
        # 'RPM': '1201',          # optional
        'axis': 'F',            # optional
        'input_feature': 'rms'  # 필수. 모델 input feature로 사용할 데이터
    }

    yaml_config = load_yaml('./model_config.yaml')

    main(yaml_config, target_config, save_figs=False, save_model=False, save_log=True)
