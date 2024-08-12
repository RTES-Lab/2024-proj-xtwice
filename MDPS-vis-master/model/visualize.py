# visualize.py
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

file = '../output/0724/0724_30204_H_1200/x.csv'  # 파일 경로 업데이트
Fs = 120
T = 1/Fs

df = pd.read_csv(file, usecols=[0, 1], names=['A', 'B'], header=None)

print(df)
# Calculate the means of each column
means = df.mean()

# Subtract the mean from each column to center the data
df_centered = df - means
print(df_centered)

# print(means)
# Compute FFT for each column and calculate amplitude and frequency
df_fft = np.fft.fft(df_centered, axis=0)
amp = np.abs(df_fft)*(2/len(df_fft))
freq = np.fft.fftfreq(len(df_fft), T)

# Plot each centered column
for column in df_centered.columns:
    plt.figure(figsize=(15, 6))
    plt.plot(df_centered.index / 120, df_centered[column])
    plt.title(f'Centered Values of {column}', size=15)
    plt.xlabel('Time[s]', size=15)
    plt.ylabel('Displacement[mm]', size=15)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.savefig(f'./{column}.png')
    plt.close()

# Function to find peaks in FFT results
def find_peaks(freq, amp, num_peaks=5):
    peak_indices = np.argsort(amp)[-num_peaks:]
    return freq[peak_indices], amp[peak_indices]

# Plot FFT results and identify peaks
for i, column in enumerate(df_centered.columns):
    plt.figure(figsize=(15, 6))
    plt.plot(freq[:len(freq)//2], amp[:len(amp)//2, i])  # Plot only the positive frequencies
    
    # Find and plot peaks
    peak_freq, peak_amp = find_peaks(freq[:len(freq)//2], amp[:len(amp)//2, i])
    plt.plot(peak_freq, peak_amp, 'ro')  # Mark peaks with red dots
    
    for j in range(len(peak_freq)):
        plt.text(peak_freq[j], peak_amp[j], f'{peak_freq[j]:.2f}Hz', fontsize=12)
    
    plt.title(f'FFT of Centered Values of {column}', size=15)
    plt.xlabel('Frequency [Hz]', size=15)
    plt.ylabel('Amplitude', size=15)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.savefig(f'./fft_{column}.png')
    plt.close()

df_centered_means = df_centered.mean()
df_centered_var = df_centered.var()
max_displacement = df_centered.max()
min_displacement = df_centered.min()

print(f"Mean Displacement:\n{df_centered_means}")
print(f"Variance of Displacement:\n{df_centered_var}")
print(f"Maximum Displacement:\n{max_displacement}")
print(f"Minimum Displacement:\n{min_displacement}")