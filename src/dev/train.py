import argparse
import config
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Callable
from tqdm import tqdm

from dev.dataloader import SpeechDataLoader
from dev.models.inceptiontime import InceptionTime

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

    args = parser.parse_args()
    return args

def train(model: Callable, train_loader: DataLoader, val_loader: DataLoader, save = True):
    optimizer = torch.optim.Adam(params = model.parameters(), lr = args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.5, patience = 3)
    criterion = nn.CrossEntropyLoss()
    model.to(config.DEVICE)
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        tk0 = tqdm(train_loader, total = len(train_loader))
        for _, (inputs, targets) in enumerate(tk0):
            optimizer.zero_grad()
            outs = model(inputs.to(config.DEVICE, dtype=torch.float))
            loss = criterion(outs, targets.to(config.DEVICE))
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            
        val_loss, val_acc = eval(model, val_loader, args)
        print(f"Epoch {epoch + 1}: Train Loss = {epoch_loss/len(train_loader)}, Val Loss = {val_loss}, Val Accuracy = {val_acc}")
        scheduler.step(val_loss)
        
    if save:
        if not config.MODEL_PATH.exists():
            config.MODEL_PATH.mkdir()
        ckpt_path = str(config.MODEL_PATH / "model.pt")
        model.save_state_dict(ckpt_path)
                    
            
def eval(model, val_loader):
    model.eval()
    val_loss = 0
    total_count = 0
    correct_count = 0
    with torch.no_grad():
        tk0 = tqdm(val_loader, total = len(val_loader))
        for inputs, targets in tk0:
            outs = model(inputs.to(config.DEVICE, dtype=torch.float))
            loss = nn.CrossEntropyLoss()(outs, targets.to(config.DEVICE))
            val_loss += loss.item()
            preds = torch.argmax(outs, dim = 1)
            total_count += len(targets)
            correct_count += (preds == targets).sum().item()
    
    return val_loss / len(val_loader), correct_count / total_count * 100
    
if __name__ == "__main__":
    args = get_argument_parser()
    batch_size = args.batch_size
    epochs_no = args.epochs
    learning_rate = args.learning_rate
    
    data_manager = SpeechDataLoader(batch_size = batch_size)
    data_manager.setup()
    train_loader, val_loader = data_manager.train_loader(), data_manager.validation_loader()
    model = InceptionTime(in_channels = config.MEL_CHANNELS,
                          sequence_len = config.SEQUENCE_LEN,
                          num_classes = len(config.LABELS))
    train(model, train_loader, val_loader)
    
## python3 src/dev/train.py -e 1 -bs 128