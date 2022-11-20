"""Utility Functions
"""
from typing import Callable

import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import signal

from config import DEVICE, EPS, FFT_OVERLAP, FFT_WINDOW, LOGGER


def save_model(model: Callable, path: str):
    """Save Model

    Args:
        model (Callable): Model
        path (str): Path to save ckpt
    """
    torch.save(model.state_dict(), path)
    LOGGER.info(f"Model saved successfully at {path}")


def load_model(model: Callable, path: str):
    """Load Model

    Args:
        model (Callable): Model
        path (str): Path to load ckpt
    """
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    LOGGER.info(f"Model {str(model)} loaded successfully from {path}")


def log_spectrograms(sample_data: np.array, sample_rate: int):
    """Plot Spectrogram

    Args:
        sample_data (np.array): Sequence of time series sound data
        sample_rate (int): Sample rate
    """

    freqs, times, spec = signal.spectrogram(
        sample_data,
        fs=sample_rate,
        window="hann",
        nperseg=FFT_WINDOW,
        noverlap=FFT_OVERLAP,
        detrend=False,
    )
    spec = np.log(spec.T.astype(np.float32) + EPS)
    plt.figure(figsize=(12, 6))
    ## Amplitude vs Time Plot
    plt.subplot(1, 2, 1)
    plt.plot(sample_data)
    plt.xlabel("Time Stamp")
    plt.ylabel("Amplitude")
    plt.title("Time Series")

    ## Spectrogram Plot after Discrete Fast-Fourier Transform
    plt.subplot(1, 2, 2)
    plt.imshow(
        spec,
        aspect="auto",
        origin="lower",
        extent=[times.min(), times.max(), freqs.min(), freqs.max()],
    )
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.title("Spectrogram - y axis is frequency spectrum")


def mel_spectrogram(sample_data: np.array, sample_rate: int, noverlap: int):
    """Plot Mel_Spectrogram

    Args:
        sample_data (np.array): Sequence of time series data
        sample_rate (int): Sample rate
        noverlap (int): Overlapping Window
    """
    ## Sample data = size (time, feature_channels)
    librosa.display.specshow(
        y=sample_data,
        x_axis="mel",
        fmax=sample_rate,
        y_axis="time",
        sr=sample_rate,
        hop_length=noverlap,
    )
