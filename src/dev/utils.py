"""Utility Functions
"""
import os
import pickle
import random
from typing import Callable, Literal

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy import signal
from sklearn.metrics import classification_report, confusion_matrix
from torchaudio.functional import edit_distance

from config import (
    DATA_PATH,
    DEVICE,
    EPS,
    FFT_OVERLAP,
    FFT_WINDOW,
    KAGGLE_LABELS,
    LABELS,
    LOGGER,
    MEL_CHANNELS,
    MODEL_PATH,
    SAMPLE_RATE,
)


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


def ff_transform(
    audio: np.array,
    fft_type: Literal["spectrogram", "mfcc"] = "mfcc",
    fft_window: int = FFT_WINDOW,
    hop_length: int = FFT_OVERLAP,
    sample_rate: int = SAMPLE_RATE,
    mel_channels: int = MEL_CHANNELS,
) -> np.array:
    """Transform from time domain to frequency domain

    Args:
        audio (np.array): Input numpy array

    Returns:
        np.array: Time Sequence of np.arrays
    """

    if fft_type == "spectrogram":
        db = librosa.stft(y=audio, n_fft=fft_window, hop_length=hop_length)
        S = librosa.amplitude_to_db(np.abs(db), ref=np.max)

    elif fft_type == "mfcc":
        power = librosa.feature.melspectrogram(
            y=audio,
            sr=sample_rate,
            n_fft=fft_window,
            hop_length=hop_length,
            n_mels=mel_channels,
        )
        S = librosa.power_to_db(power, ref=np.max)

    return S


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
    """Generate Summary Statistics:
    1. Classification Report containing Macro, Micro, Weighted Avg Precision, Recall, F1-Score
    2. Confusion Matrix

    Args:
        y_true (list): List of ground-truth labels
        y_preds (list): List of predicted labels
        path (str): Directory to save model summary artifacts`
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


def generate_pickle_data(sample_rate: int, out_path: str):
    """Generate pickle datasets
    1. Train Dataset
    2. Validation & Test datasets
    3. Inference dataset (No output labels)

    Args:
        sample_rate (int): _description_
    """

    with open(os.path.join(str(DATA_PATH), "train", "validation_list.txt"), "r") as f:
        val_paths = [file.rstrip() for file in f.readlines()]
        val_targets = [path.split("/")[0] for path in val_paths]
        val_paths = [
            os.path.join(str(DATA_PATH), "train", "audio", path) for path in val_paths
        ]

    with open(os.path.join(str(DATA_PATH), "train", "testing_list.txt"), "r") as f:
        test_paths = [file.rstrip() for file in f.readlines()]
        test_targets = [path.split("/")[0] for path in test_paths]
        test_paths = [
            os.path.join(str(DATA_PATH), "train", "audio", path) for path in test_paths
        ]

    train_paths = []
    train_targets = []

    for label in LABELS[1:]:
        filelist = os.listdir(os.path.join(str(DATA_PATH), "train", "audio", label))
        for file in filelist:
            filepath = label + "/" + file
            if filepath not in val_paths and filepath not in test_paths:
                train_paths.append(
                    os.path.join(str(DATA_PATH), "train", "audio", filepath)
                )
                train_targets.append(label)

    predict_paths = [
        os.path.join(str(DATA_PATH), "test", "audio", path)
        for path in os.listdir(os.path.join(str(DATA_PATH), "test", "audio"))
    ]

    print("Data Loaded successfully into memory")

    x_val, x_test, x_train, x_predict = [], [], [], []

    for path in val_paths:
        audio, _ = librosa.load(path, sr=sample_rate)
        x_val.append(audio)
    with open(os.path.join(out_path, "validation.pkl"), "wb") as f:
        pickle.dump((x_val, val_targets), f)

    for path in test_paths:
        audio, _ = librosa.load(path, sr=sample_rate)
        x_test.append(audio)
    with open(os.path.join(out_path, "test.pkl"), "wb") as f:
        pickle.dump((x_test, test_targets), f)

    for path in train_paths:
        audio, _ = librosa.load(path, sr=sample_rate)
        x_train.append(audio)
    with open(os.path.join(out_path, "train.pkl"), "wb") as f:
        pickle.dump((x_train, train_targets), f)

    for path in predict_paths:
        audio, _ = librosa.load(path, sr=sample_rate)
        x_predict.append(audio)
    with open(os.path.join(out_path, "predict.pkl"), "wb") as f:
        pickle.dump(x_predict, f)

    print("Data saved successfully to disk")


def generate_lexicons(artifact_path: str = str(MODEL_PATH / "meta")):
    """Generate Lexicons for CTCDecoder

    Args:
        artifact_path (str, optional): Path to dump lexicons txt file. Defaults to str(MODEL_PATH / "meta").
    """
    with open(os.path.join(artifact_path, "lexicon.txt"), "w") as f:
        f.write("# lexicons.txt\n")
        for label in LABELS:
            if label == "silence":
                label = ""
            space_split_tokens = " ".join(list(label))
            f.write(label + "\t" + space_split_tokens + " |\n")


def find_best_label(pred: str, labels: list = [""] + LABELS[1:]):
    """Find the closest label to the prediction

    Args:
        pred (str): Prediction Generated by CTC Decoding
        labels (list, optional): List of possible labels.

    Returns:
        str: best label
    """
    min_dist = float("inf")
    for label in labels:
        dist = edit_distance(pred, label)
        if dist < min_dist:
            best_label = label
            min_dist = dist

    return "silence" if best_label == "" else best_label


def convert_kaggle_label(pred: str):
    """Convert Class Labels into Kaggle templates.
    Kaggle Templates includes 12 classes:
    "yes", "no", "up", "down", "left", "right", "on",
    "off", "stop", "go", "silence", "unknown"

    Args:
        pred (str): model raw predicted label

    Returns:
        str: kaggle formatted label
    """
    return pred if pred in KAGGLE_LABELS else "unknown"
