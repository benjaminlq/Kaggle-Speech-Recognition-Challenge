import matplotlib.pyplot as plt
import numpy as np
from config import FFT_OVERLAP, FFT_WINDOW, EPS, DEVICE
import librosa
from scipy import signal
import torch

def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved successfully at {path}")

def load_model(model, path):
    model.load_state_dict(torch.load(path, map_location = DEVICE))
    print(f"Model {str(model)} loaded successfully from {path}")

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

def mel_spectrogram(sample_data: np.array, sample_rate: int, noverlap: int):
    ## Sample data = size (time, feature_channels)
    librosa.display.specshow(y = sample_data, x_axis="mel", fmax=sample_rate,
                             y_axis="time", sr=sample_rate, hop_length=noverlap)
    

    
    
