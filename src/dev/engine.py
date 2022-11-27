"""Functions for training and evaluating
"""
from typing import Callable, Tuple

import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from config import LOGGER
from dev import utils


def train_classification(
    model: Callable,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs_no: int,
    learning_rate: float,
    early_stopping: bool = False,
    patience: int = 5,
    save: bool = True,
    load: bool = False,
    debug: bool = False,
):
    """Train model as a Multiclass Classification model using CrossEntropyLoss.

    Args:
        model (Callable): Model to train
        train_loader (DataLoader): Training Dataloader
        val_loader (DataLoader): Validatation Dataloader
        epochs_no (int): No of epochs
        learning_rate (float): Learning Rate
        early_stopping (bool, optional): Whether to use EarlyStopping. Defaults to False.
        patience (int, optional): How many epochs without val_acc improvement for EarlyStopping. Defaults to 5.
        save (bool, optional): Save model. Defaults to True.
        load (bool, optional): Load model from checkpoint before training. Defaults to False.
        debug (bool, optional): Run 1 batch for debug purpose only
    """
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=3
    )
    criterion = nn.CrossEntropyLoss()

    model.to(config.DEVICE)
    LOGGER.info(f"Training models on {config.DEVICE}")

    if load:
        ckpt_path = config.MODEL_PATH / "model_ckpt" / str(model) / "model.pt"
        if not ckpt_path.exists():
            LOGGER.info(
                f"Ckpt_path for {str(model)} does not exist. Training New Model"
            )
        utils.load_model(model, ckpt_path)

    best_val_acc = 0
    patience_count = 0
    history = {"train_loss": [], "val_loss": [], "val_accuracy": []}

    for epoch in range(epochs_no):
        model.train()
        epoch_loss = 0
        tk0 = tqdm(train_loader, total=len(train_loader))
        for batch_idx, (inputs, targets) in enumerate(tk0):
            optimizer.zero_grad()
            outs = model(inputs.to(config.DEVICE, dtype=torch.float))
            loss = criterion(outs, targets.to(config.DEVICE))
            epoch_loss += loss.item()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.CLIP)
            optimizer.step()

            if (batch_idx + 1) % 100 == 0:
                LOGGER.info(
                    f"{str(model)}: Epoch {epoch + 1}, Batch {batch_idx + 1} Loss = {loss.item()}"
                )
                if debug:
                    break

        val_loss, val_acc, val_targets, val_preds = eval_classification(
            model, val_loader
        )
        LOGGER.info(
            f"{str(model)}: Epoch {epoch + 1}: Train Loss = {epoch_loss/len(train_loader)}, Val Loss = {val_loss}, Val Accuracy = {val_acc}"
        )
        scheduler.step(val_loss)

        history["train_loss"].append(epoch_loss / len(train_loader))
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)

        if val_acc > best_val_acc:
            LOGGER.info(
                f"{str(model)}: Validation Accuracy Improved from {best_val_acc} to {val_acc}"
            )
            best_val_acc = val_acc
            patience_count = 0
            if save:
                model_path = config.MODEL_PATH / "model_ckpt" / str(model)
                if debug:
                    model_path = model_path / "debug"
                if not model_path.exists():
                    model_path.mkdir()
                ckpt_path = str(model_path / "model.pt")
                utils.save_model(model, ckpt_path)
                utils.save_summary_statistic(val_targets, val_preds, str(model_path))

        else:
            LOGGER.info(
                f"{str(model)}: Validation Accuracy from epoch {epoch + 1} did not improve"
            )
            patience_count += 1
            if early_stopping and patience_count == patience:
                LOGGER.info(
                    f"{str(model)}: No val acc improvement for {patience} consecutive epochs. Early Stopped at epoch {epoch + 1}"
                )
                break

    return history


def eval_classification(model: Callable, val_loader: Callable) -> Tuple:
    """Evaluate Model

    Args:
        model (Callable): Model
        val_loader (Callable): Validation Dataloader

    Returns:
        Tuple: (val_loss, val_accuracy)
    """
    model.eval()
    val_loss = 0
    all_targets = []
    all_preds = []
    with torch.no_grad():
        tk0 = tqdm(val_loader, total=len(val_loader))
        for inputs, targets in tk0:
            outs = model(inputs.to(config.DEVICE, dtype=torch.float))
            loss = nn.CrossEntropyLoss()(outs, targets.to(config.DEVICE))
            val_loss += loss.item()
            preds = torch.argmax(outs, dim=1).detach().cpu()
            all_preds.extend(list(preds.numpy()))
            all_targets.extend(list(targets.detach().cpu().numpy()))

    score = f1_score(all_targets, all_preds, average="macro")

    return val_loss / len(val_loader), score, all_targets, all_preds


def train_translation(
    model: Callable,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs_no: int,
    learning_rate: float,
    save: bool = True,
    load: bool = False,
):
    """Train model as a Translation Seq-To_Seq model using CTCLoss.

    Args:
        model (Callable): Model to be trained
        train_loader (DataLoader): Training DataLoader
        val_loader (DataLoader): Validation Dataloader
        epochs_no (int): No of epochs
        learning_rate (float): Learning Rate
        save (bool, optional): Save model checkpoint. Defaults to True.
        load (bool, optional): Load model from checkpoint before training. Defaults to False.
    """

    # optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, factor=0.5, patience=15
    # )

    # model.to(config.DEVICE)
    # if load:
    #     ckpt_path = config.MODEL_PATH / "model_ckpt" / str(model) / "model.pt"
    #     if not ckpt_path.exist():
    #         LOGGER.info("Ckpt_path does not exist. Training New Model")
    #     model.load_state_dict(torch.load(str(ckpt_path), map_location=config.DEVICE))

    # for epoch in range(epochs_no):
    #     model.train()
    return
