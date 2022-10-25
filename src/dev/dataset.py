from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import os
import random
import glob
from typing import Optional, Callable, Union, Sequence

from scipy import signal
from scipy.fftpack import fft
from scipy.io import wavfile
from config import DATA_PATH

class SpeechDataset(Dataset):
    def __init__(self, data_paths: Sequence, targets: Sequence):
        
        self.data_paths = data_paths
        self.targets = targets
        
        self.transforms = None
        
    def __len__(self):
        return len(self.data_paths)
        
    def __getitem__(self, item):
        sequence = wavfile.read(self.data_paths[item])
        return sequence, self.targets[item]
        # spec, time, freq = signal.spectrogram(sequence, )
        # return sequence, freq, self.targets
        
class SpeechDataLoader(DataLoader):
    def __init__(
        self,
        data_dir: Union[str, Path] = DATA_PATH,
        batch_size: int = 32,
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
                    self.train_paths.append(filepath)
                    self.train_targets.append(self.label2id[label])
        
        print(len(self.train_targets))
        print(len(self.val_targets))
        print(len(self.test_targets))

    def setup(self):
        self.train_dataset = SpeechDataset()
        self.val_dataset = SpeechDataset()
        self.test_dataset = SpeechDataset()
        self.inference_dataset = SpeechDataset()
        
    def setup_silence(self):
        self.silence = None
        
    def train_loader(self):
        return DataLoader(self.train_dataset, batch_size = self.batch_size,
                          shuffle = True, drop_last = True)
        
    def validation_loader(self):
        return DataLoader(self.val_dataset, batch_size = self.batch_size,
                          shuffle = False, drop_last = False)

    def test_loader(self):
        return DataLoader(self.test_dataset, batch_size = self.batch_size,
                          shuffle = False, drop_last = False)
        
    def predict_loader(self):
        return DataLoader(self.inference_dataset, batch_size = self.batch_size,
                          shuffle = False, drop_last = False)
    
if __name__ == "__main__":
    dataloader =  SpeechDataLoader()
    