import torch
import torch.nn as nn
import torch.nn.functional as F

class UTime(nn.Module):
    def __init__(self):
        super(UTime, self).__init__()
        
    def forward(self, inputs):
        return 