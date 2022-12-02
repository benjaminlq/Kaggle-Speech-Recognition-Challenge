"""SpeechDataset Module
"""
from typing import Callable, List, Literal, Optional, Sequence, Tuple, Union

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

import config
from dev import utils


class SpeechDataset(Dataset):
    def __init__(
        self,
        data_images: List,
        load_pickle: bool = True,
        targets: Optional[Sequence] = None,
        transform: Optional[Callable] = None,
        fft_type: Literal["spectrogram", "mfcc"] = "mfcc",
        sample_rate: int = config.SAMPLE_RATE,
        padding_type: str = "linear_ramp",
        padding_length: int = config.PADDING_LENGTH,
        fft_window: int = config.FFT_WINDOW,
        fft_overlap: int = config.FFT_OVERLAP,
        mel_channels: int = config.MEL_CHANNELS,
        augmentation: bool = False,
        pitch_shift: int = 4,
        time_stretch: float = 0.1,
        random_shift: float = 0.2,
        white_noise_factor: float = 0.005,
        sequence_output: bool = False,
    ):
        """SpeechDataset Module

        Args:
            data_images (List): List of np.arrays or paths to .wav files
            load_pickle (bool, optional): If True, data images are list of np.arrays. If False, data_images are list of .wav files. Defaults to True.
            targets (Optional[Sequence], optional): Target Labels. Defaults to None.
            transform (Optional[Callable], optional): Additional Transforms & Data Augmentation. Defaults to None.
            fft_type (Literal["spectrogram";, "mfcc"], optional): Method to use for Fourier Transform to generate feature for each timestamp. If "spectrogram",
            use Discrete Short Time Fourier Transform to convert time domain to frequency domain with feature size = FFT_WINDOW // 2 + 1. If "mfcc", use Mel Frequency
            cepstral coefficient to generate features of size = MEL_CHANNELS. Defaults to "mfcc".
            sample_rate (int, optional): Sample rate of sound files. Defaults to config.SAMPLE_RATE.
            padding_type (str, optional): Type of padding. Defaults to "linear_ramp".
            padding_length (int, optional): Length of padded sequence. Defaults to config.PADDING_LENGTH.
            fft_window (int, optional): Window for FFT. Defaults to config.FFT_WINDOW.
            fft_overlap (int, optional): Window overlapping size for FFT. Defaults to config.FFT_OVERLAP.
            mel_channels (int, optional): No of MEL channels if "mfcc" is used for fft_type. Defaults to config.MEL_CHANNELS.
            augmentation (bool, optional): Data Augmentation. Defaults to False.
            pitch_shift (int, optional): Pitch Shift Augmentation. Defaults to 4.
            time_stretch (float, optional): Time Stretch Augmentation. Defaults to 0.1.
            random_shift (float, optional): Random Shift Augmentation. Defaults to 0.2.
            white_noise_factor (float, optional): White Noise Augmentation. Defaults to 0.005.
            sequence_output (bool, optional): If False, return single label index. If True, return sequence of character index. Default to False.
        """

        self.targets = targets
        self.data_images = data_images
        self.load_pickle = load_pickle
        self.transforms = transform

        ## FFT Parameters
        self.sample_rate = sample_rate
        self.padding_length = padding_length
        self.padding_type = padding_type
        self.fft_type = fft_type
        self.fft_window = fft_window
        self.fft_overlap = fft_overlap
        self.mel_channels = mel_channels

        self.augmentation = augmentation
        self.pitch_shift = pitch_shift
        self.time_stretch = time_stretch
        self.white_noise_factor = white_noise_factor
        self.random_shift = random_shift
        self.sequence_output = sequence_output

    def __len__(self):
        """
        Returns:
            int: Size of Dataset
        """
        return len(self.data_images)

    def ff_transform(self, audio: np.array) -> np.array:
        """Transform from time domain to frequency domain

        Args:
            audio (np.array): Input numpy array

        Returns:
            np.array: Time Sequence of np.arrays
        """

        if self.fft_type == "spectrogram":
            db = librosa.stft(
                y=audio, n_fft=self.fft_window, hop_length=self.fft_overlap
            )
            S = librosa.amplitude_to_db(np.abs(db), ref=np.max)

        elif self.fft_type == "mfcc":
            power = librosa.feature.melspectrogram(
                y=audio,
                sr=self.sample_rate,
                n_fft=self.fft_window,
                hop_length=self.fft_overlap,
                n_mels=self.mel_channels,
            )
            S = librosa.power_to_db(power, ref=np.max)

        return S

    def __getitem__(self, item: int) -> Union[Tuple, torch.tensor]:
        """
        Args:
            item (int): idx of item

        Returns:
            Union[Tuple, torch.tensor]: Image or (Image, label)
        """
        if self.load_pickle:
            audio = self.data_images[item]
        else:
            audio, _ = librosa.load(self.data_images[item], sr=self.sample_rate)

        if self.augmentation:
            step = self.pitch_shift * np.random.uniform()
            speed_change = np.random.uniform(
                low=1 - self.time_stretch, high=1 + self.time_stretch
            )

            audio = librosa.effects.pitch_shift(
                y=audio, sr=self.sample_rate, n_steps=step
            )
            audio = librosa.effects.time_stretch(y=audio, rate=speed_change)

            start_idx = int(
                np.random.uniform(
                    -len(audio) * self.random_shift, len(audio) * self.random_shift
                )
            )

            if start_idx >= 0:
                audio = np.r_[
                    audio[start_idx:], np.random.uniform(-0.001, 0.001, start_idx)
                ]
            else:
                audio = np.r_[
                    np.random.uniform(-0.001, 0.001, -start_idx), audio[:start_idx]
                ]

            white_noise = np.random.randn(len(audio))
            audio += white_noise * self.white_noise_factor

        if len(audio) < self.padding_length:
            audio = np.pad(
                audio, (0, self.padding_length - len(audio)), self.padding_type
            )

        audio = audio[: self.padding_length]

        ## Convert 1D data into 2D data
        # S = self.ff_transform(audio)
        S = utils.ff_transform(
            audio,
            self.fft_type,
            self.fft_window,
            self.fft_overlap,
            self.sample_rate,
            self.mel_channels,
        )

        if self.sequence_output:
            sequence = config.ID2SEQUENCE[self.targets[item]]
            padded_sequence = F.pad(
                torch.tensor(sequence, dtype=torch.int32),
                pad=(0, config.CTC_MAX_OUT_LEN - len(sequence)),
                value=-1,
            )
            return (
                torch.tensor(S)
                if self.targets is None
                else (
                    torch.tensor(S, dtype=torch.float32),
                    padded_sequence,
                    len(sequence),
                    config.LABELS[self.targets[item]],
                )
            )

        else:
            return (
                torch.tensor(S)
                if self.targets is None
                else (torch.tensor(S), self.targets[item])
            )
