import torch
import pickle
from pathlib import Path

from torch.utils.data import Dataset
import librosa
import numpy as np
from typing import Optional, Callable, Sequence, Literal, List, Union
import config


class SpeechDataset(Dataset):
    def __init__(
        self,
        # data_paths: Union[List, str],
        data_path: Union[str, Path],
        targets: Optional[Sequence] = None,
        transform: Optional[Callable] = None,
        fft_type: Literal["spectrogram", "mfcc"] = "mfcc",
        sample_rate: int = config.SAMPLE_RATE,
        padding_type : str = "linear_ramp",
        augmentation: bool = False,
        pitch_shift: int = 4,
        time_stretch: float = 0.1,
        random_shift: float = 0.2,
        white_noise_factor: float = 0.005,
        ):
        
        self.targets = targets
        # self.data_paths = data_paths
        
        with open(data_path, "rb") as f:
            data = pickle.load(f)
            if len(data) == 2:
                self.images, self.targets = data[0], data[1]
            
        self.transforms = transform
        self.sample_rate = sample_rate
        self.padding_type = padding_type
        self.fft_type = fft_type
        
        self.augmentation = augmentation
        self.pitch_shift = pitch_shift
        self.time_stretch = time_stretch
        self.white_noise_factor = white_noise_factor
        self.random_shift = random_shift
        
    def __len__(self):
        return len(self.images)
    
    def ff_transform(self, audio):
        if self.fft_type == "spectrogram":
            db = librosa.stft (y = audio, n_fft = config.FFT_WINDOW, 
                               hop_length=config.FFT_OVERLAP, center = False)
            S = librosa.amplitude_to_db (db, ref = np.max)

        elif self.fft_type == "mfcc":
            power = librosa.feature.melspectrogram (y = audio, sr = self.sample_rate,
                                                    n_fft = config.FFT_WINDOW, hop_length = config.FFT_OVERLAP, n_mels = config.MEL_CHANNELS)
            S = librosa.power_to_db (power, ref = np.max)
            
        return S
        
    def __getitem__(self, item):
        audio = self.images[item]
        
        if self.augmentation:
            step = self.pitch_shift * np.random.uniform()
            speed_change = np.random.uniform(low=1-self.time_stretch, high=1+self.time_stretch)
            
            audio = librosa.effects.pitch_shift(y = audio, sr = config.SAMPLE_RATE, n_steps = step)
            audio = librosa.effects.time_stretch(y = audio, rate = speed_change)
            
            start_idx = int(np.random.uniform(-len(audio) * self.random_shift,
                                              len(audio) * self.random_shift))
            
            if start_idx >= 0:
                audio = np.r_[audio[start_idx:], np.random.uniform(-0.001,0.001, start_idx)]
            else:
                audio = np.r_[np.random.uniform(-0.001,0.001, -start_idx), audio[:start_idx]]
            
            white_noise = np.random.randn(len(audio))
            audio += white_noise * self.white_noise_factor
        
        if len(audio) < self.sample_rate:
            audio = np.pad(audio, (0, 16000-len(audio)), self.padding_type)
        
        audio = audio[:self.sample_rate]
        
        ## Convert 1D data into 2D data
        S = self.ff_transform(audio)
            
        return (torch.tensor(S), self.targets[item]) if self.targets else torch.tensor(S)
        
    