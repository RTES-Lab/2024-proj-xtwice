# main.py

from funs.utils import *
from funs.databuilder import *
from funs.dataset import *
from funs.processing import *
from funs.model import *
from funs.trainer import *

from torch.utils.data import DataLoader

# load yaml
config = load_yaml('./model_config.yaml')

set_seed(config.seed)

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')


output_root = os.path.join(config.output_dir, config.date)

df = make_dataframe(output_root, config)

Adata, Alabel = build_from_dataframe(df, config.sample_length, config.shift, data_column='Adata', is_onehot=False)
Bdata, Blabel = build_from_dataframe(df,  config.sample_length, config.shift, data_column='Bdata', is_onehot=False)

train_dataset_A = NumpyDataset(data=Adata, label=Alabel, transform=NpToTensor(), target_transform=NpToTensor())
train_dataset_B = NumpyDataset(data=Bdata, label=Blabel, transform=NpToTensor(), target_transform=NpToTensor())

train_loader_A = DataLoader(dataset=train_dataset_A, batch_size=config.batch_size, shuffle=True, drop_last=False)
train_loader_B = DataLoader(dataset=train_dataset_B, batch_size=config.batch_size, shuffle=True, drop_last=False)

model = Simple1DCNN(num_classes=4).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss = torch.nn.CrossEntropyLoss()


train_acc_list = []
train_loss_list = []

for epoch in range(1, config.epoch + 1):
    train_loss, train_accuracy = train(model, train_loader_A, optimizer, DEVICE, loss, config.batch_size)
    train_acc_list.append(train_accuracy)
    train_loss_list.append(train_loss)
    print("[EPOCH: {}] \tTrain Loss: {:.4f}, \tTrain Accuracy: {:.4f}".format(
        epoch, train_loss, train_accuracy))
    
plot_training_curves(train_loss_list, train_acc_list, data='A')


train_acc_list = []
train_loss_list = []

for epoch in range(1, config.epoch + 1):
    train_loss, train_accuracy = train(model, train_loader_B, optimizer, DEVICE, loss, config.batch_size)
    train_acc_list.append(train_accuracy)
    train_loss_list.append(train_loss)
    print("[EPOCH: {}] \tTrain Loss: {:.4f}, \tTrain Accuracy: {:.4f}".format(
        epoch, train_loss, train_accuracy))


plot_training_curves(train_loss_list, train_acc_list, data='B')