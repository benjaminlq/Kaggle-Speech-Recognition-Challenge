import config
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Callable
from tqdm import tqdm
import warnings
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
    ):
    
    optimizer = torch.optim.Adam(params = model.parameters(), lr = learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.5, patience = 3)
    criterion = nn.CrossEntropyLoss()
    
    model.to(config.DEVICE)
    print(f"Training models on {config.DEVICE}")
    
    if load:
        ckpt_path = config.MODEL_PATH / "model_ckpt" / str(model) / "model.pt"
        if not ckpt_path.exists():
            warnings.warn("Ckpt_path does not exist. Training New Model")
        utils.load_model(model, ckpt_path)
    
    best_val_acc = 0
    patience_count = 0
    
    for epoch in range(epochs_no):
        model.train()
        epoch_loss = 0
        tk0 = tqdm(train_loader, total = len(train_loader))
        for batch_idx, (inputs, targets) in enumerate(tk0):
            optimizer.zero_grad()
            outs = model(inputs.to(config.DEVICE, dtype=torch.float))
            loss = criterion(outs, targets.to(config.DEVICE))
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            
            if (batch_idx + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}, Batch {batch_idx + 1} Loss = {loss.item()}")
            
        val_loss, val_acc = eval_classification(model, val_loader)
        print(f"Epoch {epoch + 1}: Train Loss = {epoch_loss/len(train_loader)}, Val Loss = {val_loss}, Val Accuracy = {val_acc}")
        scheduler.step(val_loss)
        
        if val_acc > best_val_acc:
            print(f"Validation Accuracy Improved from {best_val_acc} to {val_acc}")
            best_val_acc = val_acc
            patience_count = 0
            if save:
                model_path = config.MODEL_PATH / "model_ckpt" / str(model)
                if not model_path.exists():
                    model_path.mkdir()
                ckpt_path = str(model_path / "model.pt")
                utils.save_model(model, ckpt_path)
                
        else:
            print(f"Validation Accuracy from epoch {epoch + 1} did not improve")
            patience_count += 1
            if early_stopping and patience_count == patience:
                print(f"No val acc improvement for {patience} consecutive epochs. Early Stopped at epoch {epoch + 1}")
                break
                        
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
            preds = torch.argmax(outs, dim = 1).detach().cpu()
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
        