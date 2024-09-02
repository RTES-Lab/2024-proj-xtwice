# plmodule.py

import pytorch_lightning as pl
import torch

from torchmetrics.functional import accuracy

class PlModule(pl.LightningModule):
    def __init__(self, model, optimizer, criterion):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        data, label = batch
        data = data.unsqueeze(1)
        output = self(data)
        loss = self.criterion(output, label)

        prediction = torch.argmax(output, dim=1)
        # correct += prediction.eq(label.view_as(prediction)).sum().item()
        acc_tot = accuracy(prediction, label)
        acc_per_class = accuracy(prediction, label, num_classes=4, average='none')

        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_acc', acc_tot, prog_bar=True, on_step=False, on_epoch=True)

        for i, acc in enumerate(acc_per_class):
            print(f'train_acc_step_class_{i}', acc)

        return loss
    
    def evaluate(self, batch, stage=None):
        data, label = batch
        output = self(data)
        loss = self.criterion(output, label)

        prediction = torch.argmax(output, dim=1)
        # correct += prediction.eq(label.view_as(prediction)).sum().item()
        acc = accuracy(prediction, label)

        if stage:
            self.log(f'{stage}_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
            self.log(f'{stage}_acc', acc, prog_bar=True, on_step=False, on_epoch=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        return self.evaluate(batch=batch, stage="val")

    def test_step(self, batch, batch_idx):
        return self.evaluate(batch=batch, stage="test")
    
    def configure_optimizers(self):
        optimizer = self.optimizer
        return [optimizer]
    

class MetricsLogger(pl.Callback):
    def __init__(self):
        super().__init__()
        self.train_losses = []
        self.train_accuracies = []

    def on_train_epoch_end(self, trainer, pl_module):
        self.train_losses.append(trainer.callback_metrics["train_loss"].item())
        self.train_accuracies.append(trainer.callback_metrics["train_acc"].item())

    