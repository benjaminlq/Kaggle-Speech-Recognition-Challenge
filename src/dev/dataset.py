import torch

from torch.utils.data import Dataset
import librosa
import numpy as np
from typing import Optional, Callable, Sequence, Literal, List, Union
import config


class SpeechDataset(Dataset):
    def __init__(
        self,
        data_images: List,
        load_pickle: bool = True,
        targets: Optional[Sequence] = None,
        transform: Optional[Callable] = None,
        fft_type: Literal["spectrogram", "mfcc"] = "mfcc",
        sample_rate: int = config.SAMPLE_RATE,
        padding_type : str = "linear_ramp",
        padding_length: int = config.PADDING_LENGTH,
        fft_window: int = config.FFT_WINDOW,
        fft_overlap: int = config.FFT_OVERLAP,
        mel_channels: int = config.MEL_CHANNELS,
        augmentation: bool = False,
        pitch_shift: int = 4,
        time_stretch: float = 0.1,
        random_shift: float = 0.2,
        white_noise_factor: float = 0.005,
        ):
        
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
        
    def __len__(self):
        return len(self.data_images)
    
    def ff_transform(self, audio):
        if self.fft_type == "spectrogram":
            db = librosa.stft (y = audio, n_fft = self.fft_window, 
                               hop_length = self.fft_overlap)
            S = librosa.amplitude_to_db (np.abs(db), ref = np.max)

        elif self.fft_type == "mfcc":
            power = librosa.feature.melspectrogram (y = audio, sr = self.sample_rate,
                                                    n_fft = self.fft_window, hop_length = self.fft_overlap, n_mels = self.mel_channels)
            S = librosa.power_to_db (power, ref = np.max)
            
        return S
        
    def __getitem__(self, item):
        if self.load_pickle:
            audio = self.data_images[item]
        else:
            audio, _ = librosa.load(self.data_images[item], sr = self.sample_rate)
        
        if self.augmentation:
            step = self.pitch_shift * np.random.uniform()
            speed_change = np.random.uniform(low=1-self.time_stretch, high=1+self.time_stretch)
            
            audio = librosa.effects.pitch_shift(y = audio, sr = self.sample_rate, n_steps = step)
            audio = librosa.effects.time_stretch(y = audio, rate = speed_change)
            
            start_idx = int(np.random.uniform(-len(audio) * self.random_shift,
                                              len(audio) * self.random_shift))
            
            if start_idx >= 0:
                audio = np.r_[audio[start_idx:], np.random.uniform(-0.001,0.001, start_idx)]
            else:
                audio = np.r_[np.random.uniform(-0.001,0.001, -start_idx), audio[:start_idx]]
            
            white_noise = np.random.randn(len(audio))
            audio += white_noise * self.white_noise_factor
        
        if len(audio) < self.padding_length:
            audio = np.pad(audio, (0, self.padding_length-len(audio)), self.padding_type)
        
        audio = audio[:self.padding_length]
        
        ## Convert 1D data into 2D data
        S = self.ff_transform(audio)
            
        return torch.tensor(S) if self.targets is None else (torch.tensor(S), self.targets[item])