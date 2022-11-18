import config
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Callable
from tqdm import tqdm
import warnings

def train_classification(
    model: Callable,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs_no: int,
    learning_rate: float,
    save: bool = True,
    load: bool = False,
    ):
    
    optimizer = torch.optim.Adam(params = model.parameters(), lr = learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.5, patience = 3)
    criterion = nn.CrossEntropyLoss()
    
    model.to(config.DEVICE)
    if load:
        ckpt_path = config.MODEL_PATH / str(model) / "model.pt"
        if not ckpt_path.exist():
            warnings.warn("Ckpt_path does not exist. Training New Model")
        model.load_state_dict(torch.load(str(ckpt_path), map_location=config.DEVICE))
    
    for epoch in range(epochs_no):
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
            
        val_loss, val_acc = eval_classification(model, val_loader)
        print(f"Epoch {epoch + 1}: Train Loss = {epoch_loss/len(train_loader)}, Val Loss = {val_loss}, Val Accuracy = {val_acc}")
        scheduler.step(val_loss)
        
    if save:
        if not config.MODEL_PATH.exists():
            config.MODEL_PATH.mkdir()
        ckpt_path = str(config.MODEL_PATH / str(model) / "model.pt")
        model.save_state_dict(ckpt_path)
        print(f"Model Saved Successfully at {ckpt_path}")
                        
def eval_classification(model, val_loader):
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

def train_translation(
    model: Callable,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs_no: int,
    learning_rate: float,
    save: bool = True,
    load: bool = False,
    ):
    
    optimizer = torch.optim.Adam(params=model.parameters(), lr = learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.5, patience = 15)
    
    model.to(config.DEVICE)
    if load:
        ckpt_path = config.MODEL_PATH / str(model) / "model.pt"
        if not ckpt_path.exist():
            warnings.warn("Ckpt_path does not exist. Training New Model")
        model.load_state_dict(torch.load(str(ckpt_path), map_location=config.DEVICE))
        