import sys
if sys.platform == 'win32':
    import os
    os.environ['PYTHONUTF8'] = '1'
import os

import tensorflow as tf
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense

from keras.utils.vis_utils import plot_model
from funs.databuilder import make_dataframe, augment_dataframe, add_rms_peak, get_data_label
from funs.utils import set_seed, load_yaml, get_peak_rms_hist_pic

from sklearn.model_selection import KFold
import numpy as np
import scipy.stats as stats


def main(yaml_config, target_config):
    # 사용 할 데이터가 있는 디렉토리
    directory = os.path.join(yaml_config.output_dir, target_config['date'])

    # 데이터프레임 제작
    df = make_dataframe(yaml_config, rpm=None, directory=directory)
 
    # 데이터 증강
    augmented_df = augment_dataframe(df, 'z', yaml_config.sample_size, yaml_config.overlap)


    # rms, peak 값 추가
    statistics_df = add_rms_peak(augmented_df, 'z')

    # peak, rms distribution 그림 (Optional)
    get_peak_rms_hist_pic(statistics_df, './RMS_and_Peak_distribution_X.png')

    # 데이터, 라벨 얻기
    X, Y = get_data_label(statistics_df, target_config['input_data'])

    # 10-Fold Cross Validation
    kfold = KFold(n_splits=10, shuffle=True, random_state=42)
    accuracies = []
    losses = []

    for fold, (train_idx, test_idx) in enumerate(kfold.split(X)):
        print(f'{fold + 1}번째 Fold 시작!')

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = Y[train_idx], Y[test_idx]

        # 모델 (ANN)
        ANN = Sequential([
            Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(4, activation='softmax')
        ])

        # 모델 컴파일
        ANN.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # 훈련
        history = ANN.fit(
            X_train, y_train, 
            epochs=yaml_config.epochs, 
            batch_size=yaml_config.batch_size, 
            validation_data=(X_test, y_test), 
            verbose=0
        )

        # 마지막 에포크의 validation accuracy 저장
        accuracy = history.history['val_accuracy'][-1]
        loss = history.history['val_loss'][-1]

        if not np.isnan(accuracy):
            accuracies.append(accuracy)
        else:
            print(f"Warning: Fold {fold + 1} produced NaN accuracy")

        if not np.isnan(loss):
            losses.append(loss)
        else:
            print(f"Warning: Fold {fold + 1} produced NaN accuracy")

    # NaN 값이 없고 실험이 충분한 경우에만 평균과 신뢰구간 계산
    if len(accuracies) > 1:
        # 정확도 평균 및 신뢰구간 계산
        mean_accuracy = np.mean(accuracies)
        accuracy_variance = np.var(accuracies)

        if accuracy_variance > 0:
            accuracy_margin = (stats.t.interval(
                0.95, len(accuracies) - 1, loc=mean_accuracy, scale=stats.sem(accuracies)
            )[1] - mean_accuracy)
            print(f"평균 정확도: {mean_accuracy:.4f} ± {accuracy_margin:.4f}")
        else:
            print(f"평균 정확도: {mean_accuracy:.4f}")
            print("정확도 값들의 분산이 0이어서 신뢰구간을 계산할 수 없습니다.")

    # 손실값 (loss) 평균 및 신뢰구간 계산
    if len(losses) > 1:
        mean_loss = np.mean(losses)
        loss_variance = np.var(losses)

        if loss_variance > 0:
            loss_margin = (stats.t.interval(
                0.95, len(losses) - 1, loc=mean_loss, scale=stats.sem(losses)
            )[1] - mean_loss)
            print(f"평균 손실값: {mean_loss:.4f} ± {loss_margin:.4f}")
        else:
            print(f"평균 손실값: {mean_loss:.4f}")
            print("손실값들의 분산이 0이어서 신뢰구간을 계산할 수 없습니다.")
    else:
        print("Not enough valid loss values to compute confidence interval")



    # 마지막 Fold의 모델을 TensorFlow Lite 형식으로 변환
    converter = tf.lite.TFLiteConverter.from_keras_model(ANN)
    tflite_model = converter.convert()

    model_dir = yaml_config.model_dir
    model_filename = f"ANN_{target_config['input_data']}.tflite"
    model_path = os.path.join(model_dir, model_filename)

    os.makedirs(model_dir, exist_ok=True)
    with open(model_path, 'wb') as f:
        f.write(tflite_model)


if __name__ == "__main__":
    target_config = {
        'date': '1105',
        'bearing_type': '6204',
        'axis': 'F',
        'input_data': 'avg'  # 모델 입력으로 사용할 데이터. 'z', 'rms', 'peak' 중 하나 입력 가능
    }

    yaml_config = load_yaml('./model_config.yaml')

    main(yaml_config, target_config)
