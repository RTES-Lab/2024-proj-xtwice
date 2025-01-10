# DLTrainer.py

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss

from sklearn.model_selection import KFold
import numpy as np

class DLTrainer:
    def __init__(self, yaml_config):
        self.history = []
        self.yaml_config = yaml_config

    def train_step(self, model, train_loader, optimizer, criterion):
        model.train()
        train_loss = 0.0

        for inputs, labels in train_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        return train_loss

    def validate_step(self, model, val_loader, criterion):
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_y_true = []
        all_y_pred = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                y_pred = np.argmax(outputs.cpu().numpy(), axis=1)
                all_y_true.extend(labels.cpu().numpy())
                all_y_pred.extend(y_pred)

        val_loss /= len(val_loader)
        val_accuracy = correct / total
        return val_loss, val_accuracy, all_y_true, all_y_pred
    
    def test_step(self, model, test_loader, criterion):
        """
        테스트 데이터를 사용하여 모델 평가
        """
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        all_y_true = []
        all_y_pred = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                all_y_true.extend(labels.cpu().numpy())
                all_y_pred.extend(predicted.cpu().numpy())

        test_loss /= len(test_loader)
        test_accuracy = correct / total
        return test_loss, test_accuracy, all_y_true, all_y_pred

    def get_best_model(self, model, train_loader, val_loader, test_loader):

        print(f"Starting training & validation to find best model")

        best_accuracy = 0.0
        best_model_state = None
        best_history = None

        val_accuracy_list = []
        val_loss_list = []
        val_true_list = []
        val_pred_list = []

        criterion = CrossEntropyLoss()

        # 모델 초기화
        optimizer = Adam(model.parameters(), lr=self.yaml_config.lr)

        history = []

        # 학습 수행
        for epoch in range(self.yaml_config.epochs):
            train_loss = self.train_step(model, train_loader, optimizer, criterion)
            val_loss, val_accuracy, val_true, val_pred = self.validate_step(model, val_loader, criterion)

            print(f"Epoch [{epoch + 1}/{self.yaml_config.epochs}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

            history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_accuracy': val_accuracy
            })


            val_accuracy_list.append(val_accuracy)
            val_loss_list.append(val_loss)
            val_true_list.append(val_true)
            val_pred_list.append(val_pred)

            if val_accuracy == 1.0:
                if best_model_state is None or val_loss < best_history['val_loss']:
                    best_model_state = model.state_dict()
                    best_history = {
                        'epoch': epoch + 1,
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        'val_accuracy': val_accuracy
                    }
                    print(f"**New best model found! Epoch: {epoch + 1}, Val Loss: {val_loss:.4f}**")
            else:
                # 일반적인 경우, val_accuracy가 더 높은 모델을 best_model로 선택
                if val_accuracy > best_accuracy:
                    best_accuracy = val_accuracy
                    best_model_state = model.state_dict()
                    best_history = {
                        'epoch': epoch + 1,
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        'val_accuracy': val_accuracy
                    }
                    print(f"**New best model found! Epoch: {epoch + 1}, Accuracy: {val_accuracy:.4f}**")

        # 최고 성능 모델 복원
        best_model = model.get_model(n_classes=4)
        best_model.load_state_dict(best_model_state)

        test_loss, test_accuracy, test_true_list, test_pred_list = self.test_step(model, test_loader, criterion)


        print(f"Best model selection completed.")
        print(f"Best model performance - Epoch: {best_history['epoch']}, "
            f"Accuracy: {best_history['val_accuracy']:.4f}, "
            f"Loss: {best_history['val_loss']:.4f}")
        print()

        return val_accuracy_list, val_loss_list, val_true_list, val_pred_list, best_model, best_history, test_accuracy, test_loss, test_true_list, test_pred_list
