# model.py

import pandas as pd

from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.metrics import accuracy_score
import tensorflow_decision_forests as tfdf

class Models:
    def __init__(self, config):
        self.config = config
        self.model = None  # 모델 객체 (ANN 또는 Random Forest)
    
    def ann(self, input_data, activation='relu', final_activation='softmax'):
        self.model = Sequential([
            Dense(128, activation=activation, input_shape=(input_data.shape[1],)),
            Dense(64, activation=activation),
            Dense(4, activation=final_activation)
        ])
        return self.model

    def random_forest(self):
        self.model = tfdf.keras.RandomForestModel(task=tfdf.keras.Task.CLASSIFICATION)
        
        return self.model

    def train(self, X_train, y_train, X_test=None, y_test=None, rf=None):
        if isinstance(self.model, Sequential):  # ANN인 경우
            self.model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            self.model.fit(
                X_train, y_train,
                epochs=self.config.epochs,
                batch_size=self.config.batch_size,
                validation_data=(X_test, y_test)
            )
        elif isinstance(self.model, tfdf.keras.RandomForestModel):  # TensorFlow Decision Forests인 경우
            if rf is not None:
                self.model.fit(rf)  # 이미 DataFrame을 tf.data.Dataset 형식으로 변환한 상태에서 훈련
            else:
                return

            if X_test is not None and y_test is not None:
                # 예측 및 정확도 계산
                test_df = pd.DataFrame(X_test)
                test_target = pd.DataFrame(y_test)
                test_df['target'] = test_target

                test_df.columns = [str(col) for col in test_df.columns]

                # X_test 데이터를 tf.data.Dataset으로 변환
                rf_test = tfdf.keras.pd_dataframe_to_tf_dataset(test_df, label="target")

                # 예측 및 정확도 계산
                predictions = self.model.predict(rf_test)
                # predictions = [pred['class_ids'][0] if 'class_ids' in pred else pred['probabilities'].argmax() for pred in predictions]
                accuracy = accuracy_score(y_test, predictions)
                print(f"Random Forest Accuracy: {accuracy:.4f}")
        else:
            raise ValueError("Model is not defined or unsupported.")

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model is not trained yet.")
        return self.model.predict(X)

    def get_model(self):
        # 현재 설정된 모델 반환
        return self.model