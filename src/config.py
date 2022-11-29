"""Configurations files containing settings:
1. Path Directories
2. Settings for processing sound files
3. Training Parameters
4. Dictionaries to translate labels/chars to encoded integers
5. Settings for Logger
"""
import logging
import os
import sys
from pathlib import Path

import torch

from dev.models.ctcmodel import CTCModel
from dev.models.inceptiontime import InceptionTime
from dev.models.rnn import RNNModel

### Path ###
MAIN_PATH = Path(__file__).resolve().parents[1]
DATA_PATH = MAIN_PATH / "data"
MODEL_PATH = MAIN_PATH / "artifacts"

### Sound Processing ###
SAMPLE_RATE = 16000
PADDING_LENGTH = 16000
FFT_WINDOW = 512
FFT_OVERLAP = 128
EPS = 1e-10
NO_CHANNELS = (
    FFT_WINDOW // 2 + 1
)  # Max Frequency = SAMPLE_RATE / 2 https://en.wikipedia.org/wiki/Nyquist_rate
MEL_CHANNELS = 128
SEQUENCE_LEN = (PADDING_LENGTH - FFT_OVERLAP) // FFT_OVERLAP + 2

### Training Parameters ###
NO_EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 3e-4
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
NUM_WORKERS = os.cpu_count()
CLIP = 1
MEAN = 0.5
STD = 0.5

### LABELS & IDS
LABELS = [
    "silence",
    "yes",
    "no",
    "up",
    "down",
    "left",
    "right",
    "on",
    "off",
    "stop",
    "go",
    "bed",
    "bird",
    "cat",
    "dog",
    "eight",
    "five",
    "four",
    "happy",
    "house",
    "marvin",
    "nine",
    "one",
    "seven",
    "sheila",
    "six",
    "three",
    "tree",
    "two",
    "wow",
    "zero",
]

LABEL2ID = {}
ID2LABEL = {}

for idx, label in enumerate(LABELS):
    LABEL2ID[label] = idx
    ID2LABEL[idx] = label

all_chars = set()
for label in LABELS[1:]:
    for char in label:
        all_chars.add(char)

CHAR_LIST = list(all_chars)
CHAR2ID = {"$": 0}
ID2CHAR = {0: "$"}

charid = 1
for char in CHAR_LIST:
    CHAR2ID[char] = charid
    ID2CHAR[charid] = char

### Model Definition
MODEL_PARAMS = {
    "inceptiontime": {
        "instance": InceptionTime,
        "transform": "mfcc",
        "params": {
            "in_channels": MEL_CHANNELS,
            "sequence_len": SEQUENCE_LEN,
            "num_classes": len(LABELS),
        },
        "type": "classification",
    },
    "ctcmodel": {
        "instance": CTCModel,
        "transform": "mfcc",
        "params": {"vocab_size": len(CHAR_LIST), "channels_no": MEL_CHANNELS},
        "type": "translation",
    },
    "RNN": {
        "instance": RNNModel,
        "transform": "mfcc",
        "params": {
            "in_channels": MEL_CHANNELS,
            "sequence_len": SEQUENCE_LEN,
            "num_classes": len(LABELS),
            "use_attention": False,
        },
        "type": "classification",
    },
    "RNN_ATT": {
        "instance": RNNModel,
        "transform": "mfcc",
        "params": {
            "in_channels": MEL_CHANNELS,
            "sequence_len": SEQUENCE_LEN,
            "num_classes": len(LABELS),
            "use_attention": True,
        },
        "type": "classification",
    },
}

### Logging configurations
LOGGER = logging.getLogger(__name__)

stream_handler = logging.StreamHandler(sys.stdout)
file_handler = logging.FileHandler(
    filename=str(MODEL_PATH / "model_ckpt" / "logfile.log")
)

formatter = logging.Formatter("%(asctime)s:%(levelname)s: %(message)s")
file_handler.setFormatter(formatter)
# stream_handler.setFormatter(formatter)

LOGGER.setLevel(logging.INFO)
LOGGER.addHandler(stream_handler)
LOGGER.addHandler(file_handler)
