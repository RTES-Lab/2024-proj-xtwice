import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

from sklearn.model_selection import KFold
import numpy as np
from tqdm import tqdm

from sklearn.model_selection import KFold
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

class Trainer:
    def __init__(self):
        self.history = []

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

    def kfold_training(self, model, data_loader, num_epochs, num_folds=10):
        # KFold 초기화
        kfold = KFold(n_splits=num_folds, shuffle=True)
        dataset = data_loader.dataset  # 전체 데이터를 가져옴

        print(f"Starting {num_folds}-Fold Cross Validation")

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
            optimizer = Adam(model_instance.parameters(), lr=0.001)
            criterion = CrossEntropyLoss()

            # GPU 사용 가능하면 이동
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model_instance.to(device)

            for epoch in range(num_epochs):
                # Training Step
                train_loss = self.train_step(model_instance, train_loader, optimizer, criterion, device)

            val_loss, val_accuracy, y_true, y_pred = self.validate_step(model_instance, val_loader, criterion, device)

            print(f"\nEpoch [{epoch + 1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

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

