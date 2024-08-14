# main.py

from funs.utils import *
from funs.databuilder import *
from funs.dataset import *
from funs.processing import *
from funs.model import *
from funs.plmodule import *

from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

# load yaml
config = load_yaml('./model_config.yaml')

set_seed(config.seed)

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')


sticker_label = f'{config.sticker_label}data'


output_root = os.path.join(config.output_dir, config.date)

df = make_dataframe(output_root, config)

data, label = build_from_dataframe(df, config.sample_length, config.shift, data_column=sticker_label, is_onehot=False)

train_dataset = NumpyDataset(data=data, label=label, transform=NpToTensor(), target_transform=NpToTensor())

train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=False, num_workers=3)

model = Simple1DCNN(num_classes=4).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss = torch.nn.CrossEntropyLoss()


training_module = PlModule(model, optimizer, loss)

ckpt_callback = ModelCheckpoint(
    dirpath='./saved',
    filename=f"{sticker_label[0]}_{config.epoch:02d}",
    monitor = 'train_loss',
    save_top_k = 1,
    mode = 'min'
)

metrics_logger = MetricsLogger()

trainer = pl.Trainer(
    accelerator="gpu",
    max_epochs=config.epoch,
    callbacks=[ckpt_callback, metrics_logger],
    log_every_n_steps=1,
    default_root_dir="./saved",
    )

trainer.fit(training_module, train_loader)

plot_training_curves(metrics_logger.train_losses, metrics_logger.train_accuracies, data=sticker_label[0])
