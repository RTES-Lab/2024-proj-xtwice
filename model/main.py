# main.py

import sys
if sys.platform == 'win32':
    import os
    os.environ['PYTHONUTF8'] = '1'
import os
 
import tensorflow as tf
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense

from funs.databuilder import make_dataframe, augment_dataframe, add_rms_peak, get_data_label
from funs.utils import set_seed, load_yaml, get_peak_rms_hist_pic

from sklearn.model_selection import train_test_split


def main(yaml_config, target_config):
    set_seed(42)

    # 사용 할 데이터가 있는 디렉토리
    directory = os.path.join(yaml_config.output_dir, target_config['date'])

    # 데이터프레임 제작
    df = make_dataframe(yaml_config, rpm = target_config['RPM'], directory = directory)

    # 데이터 증강
    augmented_df = augment_dataframe(df, 'z', yaml_config.sample_size, yaml_config.overlap)

    # rms, peak 값 추가
    statistics_df = add_rms_peak(augmented_df, 'z')

    # peak, rms distribution 그림 (Optional)
    get_peak_rms_hist_pic(statistics_df, './RMS_and_Peak_distribution.png')

    # 데이터, 라벨 얻기
    X, Y = get_data_label(statistics_df, target_config['input_data'])

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # 모델 (ANN)
    ANN = Sequential([
        Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(64, activation='relu'),
        Dense(4, activation='softmax')  
    ])

    # 훈련
    ANN.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    ANN.fit(X_train, y_train, epochs=yaml_config.epochs, batch_size=yaml_config.batch_size, validation_data=(X_test, y_test))

    # 모델을 TensorFlow Lite 형식으로 변환
    converter = tf.lite.TFLiteConverter.from_keras_model(ANN)
    tflite_model = converter.convert()

    model_dir = yaml_config.model_dir 
    model_filename = f"ANN_{target_config['input_data']}.tflite"  
    model_path = os.path.join(model_dir, model_filename)

    os.makedirs(model_dir, exist_ok=True) 
    with open(model_path, 'wb') as f:
        f.write(tflite_model)


if __name__=="__main__":
    ########################
    # 주로 수정해야 할 부분
    target_config = {
        'date': '1105',
        'bearing_type': '6204',
        'RPM': '1200',
        'axis': 'F',
        'input_data': 'z' # 모델 입력으로 사용할 데이터. 'z', 'rms', 'peak' 중 하나 입력 가능
    }
    ########################
    yaml_config = load_yaml('./model_config.yaml')

    main(yaml_config, target_config)