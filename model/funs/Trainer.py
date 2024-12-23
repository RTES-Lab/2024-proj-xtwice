from sklearn.model_selection import KFold
import numpy as np
from tqdm import tqdm

class Trainer:
    def __init__(self, yaml_config):
        self.yaml_config = yaml_config

    def train_step(self, model, X_train, y_train, X_test, y_test):
        """
        주어진 모델을 훈련하고 검증 결과를 반환하는 함수
        """
        history = model.fit(X_train, y_train, epochs=self.yaml_config.epochs, batch_size=self.yaml_config.batch_size, 
                            validation_data=(X_test, y_test), verbose=0)

        # 검증 정확도 및 손실
        accuracy = history.history['val_accuracy'][-1]
        loss = history.history['val_loss'][-1]
        
        return accuracy, loss

    def kfold_training(self, X, Y, model, n_splits=10):
        """
        KFold Cross-Validation을 사용하여 모델 훈련과 평가를 수행하는 함수
        """
        kf = KFold(n_splits=n_splits, shuffle=True)
        accuracies = []
        losses = []
        all_y_true = []
        all_y_pred = []

        for fold, (train_index, test_index) in enumerate(tqdm(kf.split(X), total=kf.get_n_splits(), desc="Folds")):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = Y[train_index], Y[test_index]

            # 모델 정의 (get_model을 통해 모델 객체를 받아옴)
            model_instance = model.get_model(X_train)

            # 모델 훈련
            accuracy, loss = self.train_step(model_instance, X_train, y_train, X_test, y_test)

            # 예측값 및 실제값 저장
            y_pred_probs = model_instance.predict(X_test)
            y_pred = np.argmax(y_pred_probs, axis=1)
            all_y_true.append(y_test)
            all_y_pred.append(y_pred)

            if not np.isnan(accuracy) and not np.isnan(loss):
                accuracies.append(accuracy)
                losses.append(loss)
            else:
                print(f"Warning: Fold {fold+1} produced NaN accuracy or loss")

        return accuracies, losses, all_y_true, all_y_pred
