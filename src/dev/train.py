"""Training Script
"""
import argparse

import config
from config import LOGGER
from dev.dataloader import SpeechDataLoader
from dev.engine import (
    eval_classification,
    eval_translation,
    train_classification,
    train_translation,
)
from dev.inference import BeamSearchCTCDecoder, GreedyCTCDecoder


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
        "-l", "--load", help="Load Model from previous ckpt", type=str, default=False
    )

    parser.add_argument(
        "-es",
        "--earlystop",
        help="Whether EarlyStop during model training",
        type=bool,
        default=False,
    )

    parser.add_argument(
        "-p",
        "--patience",
        help="How Many epoches for Early Stopping",
        type=int,
        default=5,
    )

    parser.add_argument(
        "-db",
        "--debug",
        help="Whether to run train script in debug mode",
        type=bool,
        default=False,
    )

    parser.add_argument(
        "-d",
        "--decode",
        help="Use BeamSearch or Greedy for CTC Decoding",
        type=str,
        default="greedy",
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
    early_stopping = args.earlystop
    patience = args.patience
    debug = args.debug
    decode = args.decode

    model = config.MODEL_PARAMS[model_type]["instance"](
        **config.MODEL_PARAMS[model_type]["params"]
    )

    if config.MODEL_PARAMS[model_type]["type"] == "classification":
        data_manager = SpeechDataLoader(batch_size=batch_size, debug_mode=debug)
        data_manager.setup()
        train_loader, val_loader = (
            data_manager.train_loader(),
            data_manager.validation_loader(),
        )
        train_classification(
            model,
            train_loader,
            val_loader,
            epochs_no,
            learning_rate,
            early_stopping=early_stopping,
            patience=patience,
            load=load_model,
            debug=debug,
        )
    elif config.MODEL_PARAMS[model_type]["type"] == "translation":
        data_manager = SpeechDataLoader(
            batch_size=batch_size, sequence_output=True, debug_mode=debug
        )
        data_manager.setup()
        train_loader, val_loader = (
            data_manager.train_loader(),
            data_manager.validation_loader(),
        )
        train_translation(
            model,
            train_loader,
            val_loader,
            epochs_no,
            learning_rate,
            decode_type=decode,
            early_stopping=early_stopping,
            patience=patience,
            load=load_model,
            debug=debug,
        )

    # Evaluate result on test dataset
    test_loader = data_manager.test_loader()
    if config.MODEL_PARAMS[model_type]["type"] == "classification":
        test_loss, test_acc, _, _ = eval_classification(model, test_loader)
    else:
        decoder = GreedyCTCDecoder() if decode else BeamSearchCTCDecoder()
        test_loss, test_acc = eval_translation(model, test_loader, decoder)

    LOGGER.info(
        f"Model {str(model)} - Test Loss = {test_loss}, Test Accuracy = {test_acc}"
    )

## python3 src/dev/train.py -e 1 -bs 128 -md RNN_ATT
