from pathlib import Path

### Path ###
MAIN_PATH = Path(__file__).resolve().parents[1]
DATA_PATH = MAIN_PATH / "data"
MODEL_PATH = MAIN_PATH / "artifacts"

### Training Parameters ###
NO_EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 3e-4

