import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path
from datetime import datetime
import os
import config
from typing import Optional, Callable, Union, Literal
from dev.dataset import SpeechDataset

class SpeechDataLoader(DataLoader):
    def __init__(
        self,
        data_dir: Union[str, Path] = config.DATA_PATH,
        transform: Optional[Callable] = None,
        batch_size: int = 32,
        fft_type: Literal["spectrogram", "mfcc"] = "mfcc",
        sample_rate: int = config.SAMPLE_RATE,
        padding_type : str = "linear_ramp",
        load_npy: Optional[str] = None,
    ):
        
        self.labels = ['silence','yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go',
                       'bed', 'bird', 'cat', 'dog', 'eight', 'five', 'four', 'happy', 'house', 'marvin',
                       'nine', 'one', 'seven', 'sheila', 'six', 'three', 'tree', 'two', 'wow', 'zero']
        
        self.label2id = {}
        self.id2label = {}
        
        for idx, label in enumerate(self.labels):
            self.label2id[label] = idx
            self.id2label[idx] = label
        
        self.num_classes = len(self.labels)   
            
        self.batch_size = batch_size
        
        if isinstance(data_dir, Path):
            self.data_dir = str(data_dir)
        else:
            self.data_dir = data_dir
        
        with open(os.path.join(self.data_dir, "train", "validation_list.txt"), "r") as f:
            self.val_paths = [file.rstrip() for file in f.readlines()]
            self.val_targets = [self.label2id[path.split("/")[0]] 
                                for path in self.val_paths]
        
        with open(os.path.join(self.data_dir, "train", "testing_list.txt"), "r") as f:
            self.test_paths = [file.rstrip() for file in f.readlines()]
            self.test_targets = [self.label2id[path.split("/")[0]]
                                 for path in self.test_paths]
    
        self.train_paths = []
        self.train_targets = []
    
        for label in self.labels[1:]:
            filelist = os.listdir(os.path.join(self.data_dir, "train", "audio", label))
            for file in filelist:
                filepath = label + "/" + file
                if filepath not in self.val_paths and filepath not in self.test_paths:
                    self.train_paths.append(os.path.join(self.data_dir, "train", "audio", filepath))
                    self.train_targets.append(self.label2id[label])
        
        self.predict_paths = [os.path.join(self.data_dir, "test", "audio")]
        
        self.sample_rate = sample_rate
        self.padding_type = padding_type
        self.fft_type = fft_type
        
        ## Augmentation
        
        if transform:
            self.transforms = transform
        else:
            self.transforms = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(0.5, 0.5),
                ]
                )
            
    def setup_silence(self):
        return

    def setup(self, save = True):
        self.train_dataset = SpeechDataset(
            data_paths = self.train_paths,
            targets = self.train_targets,
            transform = self.transforms,
            fft_type = self.fft_type,
            sample_rate = self.sample_rate,
            padding_type = self.padding_type,
            augmentation = True,
            )
        
        self.val_dataset = SpeechDataset(
            data_paths = self.val_paths,
            targets = self.val_targets,
            transform = self.transforms,
            fft_type = self.fft_type,
            sample_rate = self.sample_rate,
            padding_type = self.padding_type,
            augmentation = False,
        )
        
        self.test_dataset = SpeechDataset(
            data_paths = self.test_paths,
            targets = self.test_targets,
            transform = self.transforms,
            fft_type = self.fft_type,
            sample_rate = self.sample_rate,
            padding_type = self.padding_type,
            augmentation = False,
        )
        
        self.inference_dataset = SpeechDataset(
            data_paths = self.predict_paths,
            transform = self.transforms,
            fft_type = self.fft_type,
            sample_rate = self.sample_rate,
            padding_type = self.padding_type,
            augmentation = False,
        )
        
    def train_loader(self):
        return DataLoader(self.train_dataset, batch_size = self.batch_size,
                          shuffle = True, drop_last = True, num_workers=config.NUM_WORKERS)
        
    def validation_loader(self):
        return DataLoader(self.val_dataset, batch_size = self.batch_size,
                          shuffle = False, drop_last = False, num_workers=config.NUM_WORKERS)

    def test_loader(self):
        return DataLoader(self.test_dataset, batch_size = self.batch_size,
                          shuffle = False, drop_last = False, num_workers=config.NUM_WORKERS)
        
    def predict_loader(self):
        return DataLoader(self.inference_dataset, batch_size = self.batch_size,
                          shuffle = False, drop_last = False, num_workers=config.NUM_WORKERS)
        
if __name__ == "__main__":
    start_time = datetime.now()
    dataloader =  SpeechDataLoader()
    curr_time = datetime.now()
    print("Speech Loader Time:", curr_time - start_time)
    start_time = curr_time
    dataloader.setup()
    curr_time = datetime.now()
    print("Setup Time:", curr_time - start_time)
    start_time = curr_time
    train_loader = dataloader.train_loader()
    curr_time = datetime.now()
    print("Trainloader Time:", curr_time - start_time)
    start_time = curr_time
    for inputs, targets in train_loader:
        inputs, targets = next(iter(train_loader))
        curr_time = datetime.now()
        print("Time elapsed:", curr_time - start_time)
        start_time = curr_time
    print(inputs.size(), len(targets))