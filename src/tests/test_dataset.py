"""Unit Test for Dataset module
"""
import os
import pickle

import numpy as np
import torch

import config
from dev.dataset import SpeechDataset


def test_wav_dataset():
    """Test Dataset using wav files"""
    ## Prepare Dataset
    train_images = []
    train_targets = []
    for label in config.LABELS[1:]:
        filelist = os.listdir(os.path.join(config.DATA_PATH, "train", "audio", label))
        for file in filelist:
            filepath = label + "/" + file
            train_images.append(
                os.path.join(config.DATA_PATH, "train", "audio", filepath)
            )
            train_targets.append(label)

    sample_size = 15
    sample_images = np.random.choice(train_images, sample_size)
    sample_labels = np.random.choice(train_targets, sample_size)

    ## Define some parameters for feature generation behaviour
    padding_length = 16000
    fft_window = 512
    fft_overlap = 128
    sequence_len = (padding_length - fft_overlap) // fft_overlap + 2
    mel_channels = 100
    fft_channels = 1 + fft_window // 2

    ## Test for MFCC dimension
    mel_dataset = SpeechDataset(
        data_images=sample_images,
        load_pickle=False,
        targets=sample_labels,
        fft_type="mfcc",
        padding_length=padding_length,
        fft_window=fft_window,
        fft_overlap=fft_overlap,
        mel_channels=mel_channels,
    )
    assert len(mel_dataset) == sample_size, "Dataset Size mismatches"
    sample, _ = mel_dataset[0]
    assert sample.size() == torch.Size(
        (mel_channels, sequence_len)
    ), "Input Size mismatches for MFCC"

    ## Test for STFT dimension
    spec_dataset = SpeechDataset(
        data_images=sample_images,
        load_pickle=False,
        fft_type="spectrogram",
        padding_length=padding_length,
        fft_window=fft_window,
        fft_overlap=fft_overlap,
        mel_channels=mel_channels,
    )
    sample = spec_dataset[0]
    assert sample.size() == torch.Size(
        (fft_channels, sequence_len)
    ), "Input Size mismatches for FFT"


def test_pickle_dataset():
    """Test Dataset using pickle files"""
    with open(os.path.join(config.DATA_PATH, "pickle", "test.pkl"), "rb") as f:
        test_data, test_targets = pickle.load(f)
    with open(os.path.join(config.DATA_PATH, "pickle", "predict.pkl"), "rb") as f:
        predict_data = pickle.load(f)

    padding_length = 16000
    fft_window = 512
    fft_overlap = 128
    sequence_len = (padding_length - fft_overlap) // fft_overlap + 2
    mel_channels = 100
    fft_channels = 1 + fft_window // 2

    ## Test for MFCC dimension
    mel_dataset = SpeechDataset(
        data_images=test_data,
        load_pickle=True,
        targets=test_targets,
        fft_type="mfcc",
        padding_length=padding_length,
        fft_window=fft_window,
        fft_overlap=fft_overlap,
        mel_channels=mel_channels,
    )
    sample, _ = mel_dataset[0]
    assert sample.size() == torch.Size(
        (mel_channels, sequence_len)
    ), "Input Size mismatches for MFCC"

    ## Test for STFT dimension
    spec_dataset = SpeechDataset(
        data_images=predict_data,
        load_pickle=True,
        fft_type="spectrogram",
        padding_length=padding_length,
        fft_window=fft_window,
        fft_overlap=fft_overlap,
        mel_channels=mel_channels,
    )
    sample = spec_dataset[0]
    assert sample.size() == torch.Size(
        (fft_channels, sequence_len)
    ), "Input Size mismatches for FFT"
