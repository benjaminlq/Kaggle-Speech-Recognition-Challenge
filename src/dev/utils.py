"""Utility Functions
"""
import os
import random
from typing import Callable

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy import signal
from sklearn.metrics import classification_report, confusion_matrix

from config import DEVICE, EPS, FFT_OVERLAP, FFT_WINDOW, LABELS, LOGGER


def save_model(model: Callable, path: str):
    """Save Model

    Args:
        model (Callable): Model
        path (str): Path to save ckpt
    """
    torch.save(model.state_dict(), path)
    LOGGER.info(f"Model {str(model)} saved successfully at {path}")


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


def seed_everything(seed: int = 2023):
    """Set seed for reproducability"""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def save_summary_statistic(y_true: list, y_preds: list, path: str):
    """_summary_

    Args:
        y_true (list): _description_
        y_preds (list): _description_
        path (str): _description_
    """
    with open(os.path.join(path, "classification_report.txt"), "w") as f:
        f.write(classification_report(y_true, y_preds))

    cf_matrix = confusion_matrix(y_true, y_preds)
    df_cm = pd.DataFrame(cf_matrix, index=LABELS, columns=LABELS)

    plt.figure(figsize=(12, 8))
    sns.heatmap(df_cm, annot=True, fmt="g")
    plt.title("Confusion Matrix for 31 classes of words")
    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")
    plt.savefig(os.path.join(path, "Confusion Matrix.png"))


def generate_pickle_data(sample_rate: int):
    """_summary_

    Args:
        sample_rate (int): _description_
    """
    return
