import argparse
import config

from dev.engine import train_classification, train_translation
from dev.dataloader import SpeechDataLoader

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
        default=config.NO_EPOCHS,
    )

    parser.add_argument(
        "-bs",
        "--batch_size",
        help="The number of images in a batch (default: 64)",
        type=int,
        default=config.BATCH_SIZE,
    )

    parser.add_argument(
        "-lr",
        "--learning_rate",
        help="The learning rate used for optimizer (default: 1e-4)",
        type=float,
        default=config.LEARNING_RATE,
    )
    
    parser.add_argument(
        "-md",
        "--model",
        help="Type of model instance used for training",
        type=str,
        default="inceptiontime",
    )
    
    parser.add_argument(
        "-l",
        "--load",
        help="Load Model from previous ckpt",
        type=str,
        default=False
    )

    args = parser.parse_args()
    return args


    
if __name__ == "__main__":
    args = get_argument_parser()
    batch_size = args.batch_size
    epochs_no = args.epochs
    learning_rate = args.learning_rate
    model_type = args.model
    load_model = args.load
    
    data_manager = SpeechDataLoader(batch_size = batch_size)
    data_manager.setup()
    train_loader, val_loader = data_manager.train_loader(), data_manager.validation_loader()
    # model = InceptionTime(in_channels = config.MEL_CHANNELS,
    #                       sequence_len = config.SEQUENCE_LEN,
    #                       num_classes = len(config.LABELS))
    model = config.MODEL_PARAMS[model_type]["instance"](**config.MODEL_PARAMS[model_type]["params"])
    
    if config.MODEL_PARAMS[model_type]["type"] == "classification":
        train_classification(model, train_loader, val_loader, epochs_no, learning_rate, load = load_model)
    elif config.MODEL_PARAMS[model_type]["type"] == "translation":
        train_translation(model, train_loader, val_loader, epochs_no, learning_rate, load = load_model)
        
    
## python3 src/dev/train.py -e 1 -bs 128