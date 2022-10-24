import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class CTC(nn.Module):
    def __init__(self, window):
        super(CTC,self).__init__()
        
        self.LSTM = nn.LSTM()
        self.out