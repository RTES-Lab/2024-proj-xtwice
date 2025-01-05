import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

from sklearn.model_selection import KFold
import numpy as np
from tqdm import tqdm

class DLTrainer:
    def __init__(self, yaml_config):
        self.history = []
        self.yaml_config = yaml_config

    def train_step(self, model, train_loader, optimizer, criterion, device):
        model.train()
        train_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        return train_loss

    def validate_step(self, model, val_loader, criterion, device):
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_y_true = []
        all_y_pred = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
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

    def kfold_training(self, model, data_loader, n_splits=10):
        # KFold 초기화
        kfold = KFold(n_splits=n_splits, shuffle=True)
        dataset = data_loader.dataset  # 전체 데이터를 가져옴

        print(f"Starting {n_splits}-Fold Cross Validation")

        accuracies = []
        losses = []
        all_y_true = []
        all_y_pred = []

        # Fold별 훈련
        for fold, (train_idx, val_idx) in enumerate(tqdm(kfold.split(dataset), total=kfold.get_n_splits(), desc="Folds")):

            # Train/Validation 데이터 분리
            train_subset = torch.utils.data.Subset(dataset, train_idx)
            val_subset = torch.utils.data.Subset(dataset, val_idx)

            train_loader = DataLoader(train_subset, batch_size=data_loader.batch_size, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=data_loader.batch_size, shuffle=False)

            # 모델 초기화 (각 fold마다 새로운 모델)
            model_instance = model.get_model(n_classes=4)
            optimizer = Adam(model_instance.parameters(), lr=self.yaml_config.lr)
            criterion = CrossEntropyLoss()

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model_instance.to(device)

            for epoch in range(self.yaml_config.epochs):
                # Training Step
                train_loss = self.train_step(model_instance, train_loader, optimizer, criterion, device)

            val_loss, val_accuracy, y_true, y_pred = self.validate_step(model_instance, val_loader, criterion, device)

            print(f"\nEpoch [{epoch + 1}/{self.yaml_config.epochs}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

            # 기록 저장
            self.history.append({
                'fold': fold + 1,
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_accuracy': val_accuracy
            })

            accuracies.append(val_accuracy)
            losses.append(val_loss)
            all_y_true.append(y_true)
            all_y_pred.append(y_pred)

        print("Training completed across all folds.")
        return accuracies, losses, all_y_true, all_y_pred
    
    def get_best_model(self, model, data_loader, n_splits=10):
        """최고 성능의 모델을 반환하는 메소드"""
        kfold = KFold(n_splits=n_splits, shuffle=True,)  # random_state 추가로 재현성 보장
        dataset = data_loader.dataset

        print(f"Starting {n_splits}-Fold Cross Validation to find best model")

        best_accuracy = 0.0
        best_model_state = None
        best_fold_history = None

        accuracies = []
        losses = []
        all_y_true = []
        all_y_pred = []

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        criterion = CrossEntropyLoss()

        # Fold별 훈련
        for fold, (train_idx, val_idx) in enumerate(tqdm(kfold.split(dataset), total=kfold.get_n_splits())):
            # Train/Validation 데이터 분리
            train_subset = torch.utils.data.Subset(dataset, train_idx)
            val_subset = torch.utils.data.Subset(dataset, val_idx)

            train_loader = DataLoader(train_subset, batch_size=data_loader.batch_size, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=data_loader.batch_size, shuffle=False)

            # 모델 초기화
            model_instance = model.get_model(n_classes=4)
            model_instance.to(device)
            optimizer = Adam(model_instance.parameters(), lr=self.yaml_config.lr)

            fold_history = []

            # 학습 수행
            for epoch in range(self.yaml_config.epochs):
                train_loss = self.train_step(model_instance, train_loader, optimizer, criterion, device)

            # 검증 단계
            val_loss, val_accuracy, y_true, y_pred = self.validate_step(model_instance, val_loader, criterion, device)

            print(f"\nEpoch [{epoch + 1}/{self.yaml_config.epochs}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

            fold_history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_accuracy': val_accuracy
            })

            # 최고 성능 모델 갱신
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
                print(f"\nNew best model found! Fold: {fold + 1}, Epoch: {epoch + 1}, Accuracy: {val_accuracy:.4f}")

            accuracies.append(val_accuracy)
            losses.append(val_loss)
            all_y_true.append(y_true)
            all_y_pred.append(y_pred)

        # 최고 성능 모델 복원
        best_model = model.get_model(n_classes=4)
        best_model.load_state_dict(best_model_state)

        print(f"\nBest model selection completed.")
        print(f"Best model performance - Fold: {best_fold_history['fold']}, "
            f"Epoch: {best_fold_history['epoch']}, "
            f"Accuracy: {best_fold_history['val_accuracy']:.4f}")

        return accuracies, losses, all_y_true, all_y_pred, best_model, best_fold_history


