import argparse
import config

from dev.models.inceptiontime import InceptionTime
from dev.dataset import SpeechDataLoader

def get_argument_parser():
    """
    Argument parser which returns the options which the user inputted.
    Arguments:
    - Model
    - No of epochs
    - Learning rate
    - Batch size

    Returns:
        argparse.ArgumentParser().parse_args()
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-e",
        "--epochs",
        help="How many epochs you need to run (default: 10)",
        type=int,
        default=10,
    )

    parser.add_argument(
        "-bs",
        "--batch_size",
        help="The number of images in a batch (default: 64)",
        type=int,
        default=32,
    )

    parser.add_argument(
        "-lr",
        "--learning_rate",
        help="The learning rate used for optimizer (default: 1e-4)",
        type=float,
        default=1e-4,
    )

    args = parser.parse_args()
    return args

def train(model, train_loader, val_loader):
    
            
def eval(model, val_loader):
