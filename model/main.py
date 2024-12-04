# main.py

import os

import tensorflow as tf
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import KFold
import scipy.stats as stats

from tqdm import tqdm

from funs.databuilder import make_dataframe, augment_dataframe, add_statistics, get_data_label
from funs.utils import set_seed, load_yaml, get_dir_list
from funs.draw import get_stat_hist_pic

import numpy as np

def main(yaml_config, target_config, save_figs=True, save_model=True):
    set_seed(yaml_config.seed)

    directory = os.path.join(yaml_config.output_dir, target_config['date'])
    directory = get_dir_list(directory)

    # 데이터프레임 제작
    df = make_dataframe(yaml_config, directory)

    # 데이터 증강
    augmented_df = augment_dataframe(df, 'z', yaml_config.sample_size, yaml_config.overlap)

    # 통계값 값 추가
    statistics_df = add_statistics(augmented_df, 'z')

    if save_figs:
        # peak, rms distribution 그림 (Optional)
        get_stat_hist_pic(statistics_df, f'./RMS_and_Peak_distribution.png')

    # 데이터, 라벨 얻기
    X, Y = get_data_label(statistics_df, target_config['input_feature'])
    print(f'input feature: {target_config["input_feature"]}')

    # 10-fold Cross Validation
    kf = KFold(n_splits=10, shuffle=True, random_state=yaml_config.seed)
    accuracies = []
    losses = []

    for fold, (train_index, test_index) in enumerate(tqdm(kf.split(X), total=kf.get_n_splits(), desc="Folds")):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]

        # 모델 정의
        ANN = Sequential([
            Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(4, activation='softmax')
        ])

        ANN.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # 모델 학습
        history = ANN.fit(X_train, y_train, epochs=yaml_config.epochs, batch_size=yaml_config.batch_size, validation_data=(X_test, y_test), verbose=0)

        # 검증 정확도 및 손실
        accuracy = history.history['val_accuracy'][-1]
        loss = history.history['val_loss'][-1]

        if not np.isnan(accuracy) and not np.isnan(loss):
            accuracies.append(accuracy)
            losses.append(loss)
        else:
            print(f"Warning: Fold {fold+1} produced NaN accuracy or loss")

    # 결과 출력
    if len(accuracies) > 1:
        mean_accuracy = np.mean(accuracies)
        accuracy_variance = np.var(accuracies)
        mean_loss = np.mean(losses)
        loss_variance = np.var(losses)

        # 신뢰구간 계산
        if accuracy_variance > 0:
            accuracy_confidence_interval = stats.t.interval(0.95, len(accuracies)-1, loc=mean_accuracy, scale=stats.sem(accuracies))
            print(f"정확도: {mean_accuracy:.4f} ± {accuracy_confidence_interval[1] - mean_accuracy:.4f}")
        else:
            print(f"정확도: {mean_accuracy:.4f} (변동이 없어 신뢰구간을 계산할 수 없습니다.)")

        if loss_variance > 0:
            loss_confidence_interval = stats.t.interval(0.95, len(losses)-1, loc=mean_loss, scale=stats.sem(losses))
            print(f"손실: {mean_loss:.4f} ± {loss_confidence_interval[1] - mean_loss:.4f}")
        else:
            print(f"손실: {mean_loss:.4f} (변동이 없어 신뢰구간을 계산할 수 없습니다.)")
    else:
        print("Not enough valid accuracy or loss values to compute confidence interval")

    if save_model:
        converter = tf.lite.TFLiteConverter.from_keras_model(ANN)
        tflite_model = converter.convert()

        model_dir = yaml_config.model_dir
        model_filename = f"ANN_{target_config['input_feature']}.tflite"
        model_path = os.path.join(model_dir, model_filename)

        os.makedirs(model_dir, exist_ok=True)
        with open(model_path, 'wb') as f:
            f.write(tflite_model)


if __name__ == "__main__":
    target_config = {
        'date': '1105',         # 필수
        # 'bearing_type': '6204', # optional
        # 'RPM': '1201',          # optional
        # 'axis': 'F',            # optional
        'input_feature': 'z'  # 필수. 모델 input feature로 사용할 데이터
    }

    yaml_config = load_yaml('./model_config.yaml')

    main(yaml_config, target_config, save_figs=False, save_model=False)
