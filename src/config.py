from pathlib import Path
import torch
import os

### Path ###
MAIN_PATH = Path(__file__).resolve().parents[1]
DATA_PATH = MAIN_PATH / "data"
MODEL_PATH = MAIN_PATH / "artifacts"

### Sound Processing ###
SAMPLE_RATE = 16000
FFT_WINDOW = 512
FFT_OVERLAP = 128
EPS = 1e-10
NO_CHANNELS = FFT_WINDOW // 2 + 1  ### Max Frequency = SAMPLE_RATE / 2 https://en.wikipedia.org/wiki/Nyquist_rate
MEL_CHANNELS = 128
SEQUENCE_LEN = (SAMPLE_RATE - FFT_OVERLAP) // FFT_OVERLAP + 2

### Training Parameters ###
NO_EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 3e-4
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
NUM_WORKERS = os.cpu_count()