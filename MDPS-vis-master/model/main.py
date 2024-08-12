from dfb.download import *
from dfb.databuilder import *
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
from dfb.dataset import *
from dfb.trainmodule import *


def find_peaks(freq, amp, num_peaks=5):
    peak_indices = np.argsort(amp)[-num_peaks:]
    return freq[peak_indices], amp[peak_indices]

class NumpyDataset(Dataset):
    def __init__(self, data, label, transform=None, target_transform=None):
        self.data = data
        self.label = label
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        x = np.array(self.data[idx, :]).astype("float32")
        t = np.array(self.label[idx]).astype("int64")

        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            t = self.target_transform(t)

        return x, t
    
class NpToTensor:
    """numpy array를 텐서로 변환"""
    def __call__(self, x):
        return torch.from_numpy(x)
    
    
def get_dataloader(
        dataset: NumpyDataset, batch_size: int, 
        shuffle: bool, num_workers: int = 1):

    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True, drop_last=False)




    

visudalize = False

thrs = 1000
batch_size = 64

df = {}
df['Adata'] = []
df['Bdata'] = []
df["fault_type"] = []
df['axis'] = []
df['label'] = []
output_dir = '../output'
date = '0724'
root = os.path.join(output_dir, date)


label_map = {
    'H' : 0,
    'B' : 1,
    'IR': 2,
    'OR': 3,
}


for fault_type, label in label_map.items():
    fault_dir = os.path.join(root, f'{date}_30204_{fault_type}_1200')
    for axis in ['x.csv', 'z.csv']:
        fault_file = os.path.join(fault_dir, axis)
        data = pd.read_csv(fault_file, header=None)
        Adata = data.iloc[:thrs, 0].values.flatten()  # 첫 번째 열 읽어 1차원 리스트로 변환
        Bdata = data.iloc[:thrs, 1].values.flatten() # 두 번째 열 읽어 1차원 리스트로 변환
        df['Adata'].append(Adata)
        df['Bdata'].append(Bdata)
        df['fault_type'].append(fault_type)
        df['axis'].append(axis[0])
        df['label'].append(label)

test_df = pd.DataFrame(df)

for data in ['Adata', 'Bdata']:
    test_df[data] = test_df[data].apply(lambda x: x - np.mean(x))

num_classes = test_df["label"].max() - test_df["label"].min() + 1
num_data = test_df.shape[0]


Adata = np.array(test_df['Adata'].tolist())
Bdata = np.array(test_df['Bdata'].tolist())
label = np.array(test_df['label'].tolist())

train_Adataset = NumpyDataset(Adata, label=label, transform=NpToTensor(), target_transform=NpToTensor())
train_Bdataset = NumpyDataset(Bdata, label=label, transform=NpToTensor(), target_transform=NpToTensor())

train_loader_A = get_dataloader(dataset=train_Adataset,
                              batch_size=batch_size,
                              shuffle=True)
train_loader_B = get_dataloader(dataset=train_Bdataset,
                              batch_size=batch_size,
                              shuffle=True)
model_name = "wdcnn"

# print(train_loader_A.dataset[0])
model = WDCNN(n_classes=num_classes)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss = torch.nn.CrossEntropyLoss()

training_module = PlModule(model, optimizer, loss, True)

# callback = pl.callbacks.ModelCheckpoint(monitor="val_loss",
#                                         dirpath=f"./logs/best_model",
#                                         filename=f"model",
#                                         save_top_k=1,
#                                         mode="min")

trainer = pl.Trainer(
    accelerator="gpu",  # GPU를 사용할 경우 'gpu'로 설정
    devices=[0],  # 사용할 GPU 장치 목록
    max_epochs=10,
)

result = trainer.fit(model=training_module,
                     train_dataloaders=train_loader_A,)


# training_module.load_from_checkpoint(f"./model.ckpt",
#                                      model=model, optimizer=optimizer,
#                                      loss_fn=loss)
# result = trainer.test(model=training_module, dataloaders=train_loader_A)



if visudalize:
    visudalize_ex = test_df['Adata'][0]
    plt.figure(figsize=(15, 6))
    plt.plot(visudalize_ex)
    plt.title(f'Centered Values', size=15)
    plt.xlabel('Time[s]', size=15)
    plt.ylabel('Displacement[mm]', size=15)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.savefig(f'./visudalize.png')
    plt.close()

    Fs = 120
    T = 1/Fs    
    df_fft = test_df['Adata'][0]
    df_fft = np.fft.fft(df_fft, axis=0)
    amp = np.abs(df_fft)*(2/len(df_fft))
    freq = np.fft.fftfreq(len(df_fft), T)

    plt.figure(figsize=(15, 6))
    plt.plot(freq[:len(freq)//2], amp[:len(amp)//2])  # Plot only the positive frequencies
    
    # Find and plot peaks
    peak_freq, peak_amp = find_peaks(freq[:len(freq)//2], amp[:len(amp)//2])
    plt.plot(peak_freq, peak_amp, 'ro')  # Mark peaks with red dots
    
    for j in range(len(peak_freq)):
        plt.text(peak_freq[j], peak_amp[j], f'{peak_freq[j]:.2f}Hz', fontsize=12)
    
    plt.title(f'FFT of Centered Values', size=15)
    plt.xlabel('Frequency [Hz]', size=15)
    plt.ylabel('Amplitude', size=15)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.savefig(f'./visudalize_fft.png')
    plt.close()

    print('Visualize complete.')

    