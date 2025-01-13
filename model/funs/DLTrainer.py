# DLTrainer.py

import os
import datetime
import random

import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.mobile_optimizer import optimize_for_mobile

import numpy as np

from sklearn.metrics import classification_report
from sklearn.model_selection import KFold


class DLTrainer:
    def __init__(self, yaml_config):
        self.history = []
        self.yaml_config = yaml_config

    def train_step(self, model, train_loader, optimizer, criterion):
        model.train()
        train_loss = 0.0

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
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
                all_y_true.append(labels.cpu().numpy())
                all_y_pred.append(y_pred)

        val_loss /= len(val_loader)
        val_accuracy = correct / total
        return val_loss, val_accuracy, all_y_true, all_y_pred
    
    def get_best_model(self, model, data_loader, n_splits=10):
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=self.yaml_config.seed)  
        dataset = data_loader.dataset

        print(f"Starting {n_splits}-Fold Cross Validation to find best model")

        best_accuracy = 0.0
        best_model_state = None
        best_fold_history = None

        accuracies = []
        losses = []
        all_y_true = []
        all_y_pred = []

        criterion = CrossEntropyLoss()

        # Fold별 훈련
        for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
            train_subset = torch.utils.data.Subset(dataset, train_idx)
            val_subset = torch.utils.data.Subset(dataset, val_idx)

            train_loader = DataLoader(train_subset, batch_size=data_loader.batch_size, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=data_loader.batch_size, shuffle=False)

            # 모델 초기화
            model_instance = model.get_model(n_classes=4)
            optimizer = Adam(model_instance.parameters(), lr=self.yaml_config.lr)

            fold_history = []

            # 학습 수행
            for epoch in range(self.yaml_config.epochs):
                train_loss = self.train_step(model_instance, train_loader, optimizer, criterion)

            # 검증 단계
            val_loss, val_accuracy, y_true, y_pred = self.validate_step(model_instance, val_loader, criterion)

            print(f"Epoch [{epoch + 1}/{self.yaml_config.epochs}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

            fold_history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_accuracy': val_accuracy
            })

            if val_accuracy == 1.0:
                if best_model_state is None or val_loss < best_fold_history['val_loss']:
                    best_model_state = model_instance.state_dict()
                    best_fold_history = {
                        'fold': fold + 1,
                        'epoch': epoch + 1,
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        'val_accuracy': val_accuracy
                    }
                    print(f"**New best model found! Fold: {fold + 1}, Epoch: {epoch + 1}, Val Loss: {val_loss:.4f}**")
            else:
                # 일반적인 경우, val_accuracy가 더 높은 모델을 best_model로 선택
                if val_accuracy > best_accuracy:
                    best_accuracy = val_accuracy
                    best_model_state = model_instance.state_dict()
                    best_fold_history = {
                        'fold': fold + 1,
                        'epoch': epoch + 1,
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        'val_accuracy': val_accuracy
                    }
                    print(f"**New best model found! Fold: {fold + 1}, Epoch: {epoch + 1}, Accuracy: {val_accuracy:.4f}**")

            accuracies.append(val_accuracy)
            losses.append(val_loss)
            all_y_true.append(y_true)
            all_y_pred.append(y_pred)

            # 모델 저장 (각 Fold마다)
            model_name = f'Fwdcnn{fold + 1}'
            tmp_model_save_path = os.path.join(self.yaml_config.model_dir, f'{model_name}.pth')
            torch.save(best_model_state, tmp_model_save_path)

            # ptl 모델 저장 (각 Fold마다)
            best_model = model.get_model(n_classes=4)
            best_model.load_state_dict(best_model_state)

            model_save_path = os.path.join(self.yaml_config.model_dir, f'{model_name}.ptl')
            scripted_model = torch.jit.script(best_model)
            optimized_scripted_module = optimize_for_mobile(scripted_model)
            optimized_scripted_module._save_for_lite_interpreter(model_save_path)
            print(f"모델이 {model_save_path}에 저장되었습니다.")

        # 최고 성능 모델 복원
        best_model = model.get_model(n_classes=4)
        best_model.load_state_dict(best_model_state)

        print(f"Best model selection completed.")
        print(f"Best model performance - Fold: {best_fold_history['fold']}, "
            f"Epoch: {best_fold_history['epoch']}, "
            f"Accuracy: {best_fold_history['val_accuracy']:.4f}")
        print()

        return accuracies, losses, all_y_true, all_y_pred, best_model, best_fold_history