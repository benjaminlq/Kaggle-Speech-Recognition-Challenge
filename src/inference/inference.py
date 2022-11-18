import torch
import config
import librosa
import numpy as np
from pathlib import Path
from typing import Union, Callable, Sequence

def inference(audio: Union[str, Path, Sequence], model: Callable):
    if isinstance(input, str) or isinstance(input, Path):
        audio, _ = librosa.load(audio, sr = config.SAMPLE_RATE)
    
    if len(audio) < config.SEQUENCE_LEN:
        audio = np.pad(audio, (0, config.SEQUENCE_LEN-len(audio)), "linear_ramp")
    else:
        audio = audio[:config.SEQUENCE_LEN]
        
    output = model(input)
    label = config.ID2LABEL[output]
    return label

