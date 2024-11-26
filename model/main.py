import os
 
import tensorflow as tf
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense

from funs.databuilder import make_dataframe, augment_dataframe, add_rms_peak, get_data_label
from funs.utils import set_seed, load_yaml, get_peak_rms_hist

from sklearn.model_selection import train_test_split

def main(yaml_config, target_config):
    set_seed(42)

    directory = os.path.join(yaml_config.output_dir, target_config['date'])

    df = make_dataframe(yaml_config, rpm = target_config['RPM'], directory = directory)
    augmented_df = augment_dataframe(df, 'z', yaml_config.sample_size, yaml_config.overlap)
    statistics_df = add_rms_peak(augmented_df, 'z')

    # peak, rms distribution 그림 
    get_peak_rms_hist(statistics_df, './RMS_and_Peak_distribution.png')


    # X, Y = get_data_label(statistics_df, 'z')
    X, Y = get_data_label(statistics_df, 'rms')
    # X, Y = get_data_label(statistics_df, 'paek')


    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    ANN = Sequential([
        Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(64, activation='relu'),
        Dense(4, activation='softmax')  
    ])

    ANN.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    ANN.fit(X_train, y_train, epochs=yaml_config.epochs, batch_size=yaml_config.batch_size, validation_data=(X_test, y_test))

    # 모델을 TensorFlow Lite 형식으로 변환
    converter = tf.lite.TFLiteConverter.from_keras_model(ANN)
    tflite_model = converter.convert()

    # .tflite 파일 저장
    with open('ANN.tflite', 'wb') as f:
        f.write(tflite_model)

if __name__=="__main__":

    ########################
    # 주로 수정해야 할 부분
    target_config = {
        'date': '1105',
        'bearing_type': '6204',
        'RPM': '1200',
        'axis': 'F'
    }
    ########################
    yaml_config = load_yaml('./model_config.yaml')

    main(yaml_config, target_config)