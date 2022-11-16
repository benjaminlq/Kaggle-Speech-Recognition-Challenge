import matplotlib.pyplot as plt
import numpy as np
from typing import Union
from config import FFT_OVERLAP, FFT_WINDOW, EPS, SAMPLE_RATE
import librosa
from scipy import signal
from pathlib import Path
import pickle

def log_spectrograms(sample_data: np.array, sample_rate: int):
    
    freqs, times, spec = signal.spectrogram(sample_data, fs = sample_rate, window = "hann",
                                            nperseg = FFT_WINDOW, noverlap = FFT_OVERLAP, detrend = False) 
    spec = np.log(spec.T.astype(np.float32) + EPS)
    plt.figure(figsize = (12,6))
    ## Amplitude vs Time Plot
    plt.subplot(1,2,1)
    plt.plot(sample_data)
    plt.xlabel("Time Stamp")
    plt.ylabel("Amplitude")
    plt.title("Time Series")

    ## Spectrogram Plot after Discrete Fast-Fourier Transform
    plt.subplot(1,2,2)
    plt.imshow(spec, aspect='auto', origin='lower', 
            extent=[times.min(), times.max(), freqs.min(), freqs.max()])
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.title("Spectrogram - y axis is frequency spectrum")
    
# def meltrogram(sample_data: np.array):
#     x = librosa.stft(sample_data, )

def mel_spectrogram(sample_data: np.array, sample_rate: int, noverlap: int):
    ## Sample data = size (time, feature_channels)
    librosa.display.specshow(y = sample_data, x_axis="mel", fmax=sample_rate,
                             y_axis="time", sr=sample_rate, hop_length=noverlap)
    
    
