from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense

class ANN:
    def __init__(self):
        self.model = None

    def get_model(self, X_train):
        # 새 모델 생성
        self.model = Sequential([
            Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
            Dense(128, activation='relu'),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(4, activation='softmax')
        ])

        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        return self.model
