# Trainer.py

import numpy as np

class Trainer:
    def __init__(self, yaml_config):
        self.yaml_config = yaml_config

    def train_step(self, model, X_train, y_train, X_val, y_val):
        """
        주어진 모델을 훈련하고 검증 결과를 반환하는 함수
        """
        history = model.fit(X_train, y_train, epochs=self.yaml_config.epochs, batch_size=self.yaml_config.batch_size, 
                            validation_data=(X_val, y_val), verbose=0)

        # 검증 정확도 및 손실
        val_accuracy = history.history['val_accuracy'][-1]
        val_loss = history.history['val_loss'][-1]
        
        return val_accuracy, val_loss, history
    
    def test_model(self, model, X_test, y_test):
        """
        테스트 데이터를 사용하여 모델 평가
        """
        test_metrics = model.evaluate(X_test, y_test, verbose=0)
        test_loss = test_metrics[0]  
        test_accuracy = test_metrics[1]      
        return test_accuracy, test_loss

    def get_best_model(self, model, X_train, y_train, X_val, y_val, X_test, y_test):

        print(f"\nStarting training & validation to find best model!")
        model = model.get_model(X_train)
        
        val_true_list = []
        val_pred_list = []
        test_true_list = []
        test_pred_list = []

        val_accuracy, val_loss, _ = self.train_step(model, X_train, y_train, X_val, y_val)
        test_accuracy, test_loss = self.test_model(model, X_test, y_test)

        test_pred_probs = model.predict(X_test)
        test_pred = np.argmax(test_pred_probs, axis=1)
        test_true_list.append(y_test)
        test_pred_list.append(test_pred)

        val_pred_probs = model.predict(X_val)
        val_pred = np.argmax(val_pred_probs, axis=1)
        val_true_list.append(y_val)
        val_pred_list.append(val_pred)

        return val_accuracy, val_loss, test_accuracy, test_loss, test_true_list, test_pred_list, val_true_list, val_pred_list

        