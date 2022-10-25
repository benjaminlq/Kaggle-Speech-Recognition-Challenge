from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

class InceptionBlock(nn.Module):
    def __init__(
        self,
        in_channels: int = 85,
        num_filters: int = 32,
        kernel_size: List = [9, 19, 39],
        bottleneck_channels: int = 32,
    ):
        super(InceptionBlock, self).__init__()
        
        if in_channels == 1:
            self.bottleneck = self.passthrough
            bottleneck_channels = 1
        else:
            self.bottleneck = nn.Conv1d(in_channels, bottleneck_channels,
                                        kernel_size = 1, stride = 1, bias = False)
            
        self.conv1 = nn.Conv1d(in_channels = bottleneck_channels, out_channels = num_filters,
                               kernel_size = kernel_size[0], padding = kernel_size[0] // 2, bias = False)
        self.conv2 = nn.Conv1d(in_channels = bottleneck_channels, out_channels = num_filters,
                               kernel_size = kernel_size[1], padding = kernel_size[1] // 2, bias = False)
        self.conv3 = nn.Conv1d(in_channels = bottleneck_channels, out_channels = num_filters,
                               kernel_size = kernel_size[2], padding = kernel_size[2] // 2, bias = False)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride = 1, padding = 1)
        self.conv4 = nn.Conv1d(in_channels = in_channels, out_channels = num_filters,
                               kernel_size = 1, padding = 0)  
        self.batchnorm = nn.BatchNorm1d(num_features = num_filters * 4)
        
    def forward(self, inputs):
        ## Input dimension = (bs, in_channels, seq_len)
        x = F.relu(self.bottleneck(inputs))  ## (bs, bottleneck_channels, seq_len)
        x1 = F.relu(self.conv1(x)) ## (bs, num_filters, seq_len)
        x2 = F.relu(self.conv2(x)) ## (bs, num_filters, seq_len)
        x3 = F.relu(self.conv3(x)) ## (bs, num_filters, seq_len)
        x4 = F.relu(self.conv4(self.maxpool(inputs))) ## (bs, num_filters, seq_len)
        x_cat = torch.cat((x1, x2, x3, x4), axis = 1) ## (bs, num_filters * 4, seq_len)
        return F.relu(self.batchnorm(x_cat))
     
    def passthrough(self, inputs):
        return inputs
        
class Inception_ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int = 85,
        num_filters: int = 32,
        kernel_size: List = [9, 19, 39],
        bottleneck_channels: int = 32,
        residual: bool = True,
    ):
        
        super(Inception_ResidualBlock, self).__init__()
        self.inception_block1 = InceptionBlock(in_channels=in_channels, num_filters=num_filters,
                                               kernel_size=kernel_size, bottleneck_channels=bottleneck_channels)
        self.inception_block2 = InceptionBlock(in_channels=num_filters*4, num_filters=num_filters,
                                               kernel_size=kernel_size, bottleneck_channels=bottleneck_channels)
        self.inception_block3 = InceptionBlock(in_channels=num_filters*4, num_filters=num_filters,
                                               kernel_size=kernel_size, bottleneck_channels=bottleneck_channels)
        
        self.residual = residual
        self.res = nn.Sequential(nn.Conv1d(in_channels = in_channels, out_channels = num_filters * 4,
                                  kernel_size = 1, stride = 1, padding = 0),
                                 nn.BatchNorm1d(num_features = num_filters * 4))
        
    def forward(self, inputs):
        ## Inputs Dimension: (bs, in_channels, seq_len)
        x = self.inception_block1(inputs) ## (bs, num_filters * 4, seq_len)
        x = self.inception_block2(x) ## (bs, num_filters * 4, seq_len)
        x = self.inception_block3(x) ## (bs, num_filters * 4, seq_len)
        if self.residual:
            x_res = self.res(inputs) ## (bs, num_filters * 4, seq_len)
            return F.relu(x + x_res) ## (bs, num_filters * 4, seq_len)
        else:
            return x ## (bs, num_filters * 4, seq_len)
        
class InceptionTime(nn.Module):
    def __init__(
        self,
        in_channels: int,
        sequence_len: int,
        num_classes: int,
        num_filters: int = 32,
        kernel_size: List = [9, 19, 39],
        bottleneck_channels: int = 32,
        residual: bool = True,
    ):
        super(InceptionTime, self).__init__()
        self.inception_resblock1 = Inception_ResidualBlock(in_channels = in_channels, num_filters=num_filters,
                                                           kernel_size=kernel_size, bottleneck_channels=bottleneck_channels, residual = residual)
        self.inception_resblock2 = Inception_ResidualBlock(in_channels = num_filters*4, num_filters=num_filters,
                                                           kernel_size=kernel_size, bottleneck_channels=bottleneck_channels, residual = residual)
        self.inception_resblock3 = Inception_ResidualBlock(in_channels = num_filters*4, num_filters=num_filters,
                                                           kernel_size=kernel_size, bottleneck_channels=bottleneck_channels, residual = residual)
        self.inception_block_out = InceptionBlock(in_channels = num_filters*4, num_filters=num_filters,
                                                  kernel_size=kernel_size, bottleneck_channels=bottleneck_channels)
        self.avgpool = nn.AvgPool1d(kernel_size = sequence_len)
        self.linear = nn.Linear(num_filters * 4, num_classes)
        self.flatten = nn.Flatten()
        
    def forward(self, inputs):
        ## Inputs Dimension = (bs, in_channels, seq_len)
        x = self.inception_resblock1(inputs) ## (bs, num_filters * 4, seq_len)
        x = self.inception_resblock2(x) ## (bs, num_filters * 4, seq_len)
        x = self.inception_resblock3(x) ## (bs, num_filters * 4, seq_len)
        x = self.inception_block_out(x) ## (bs, num_filters * 4, seq_len)
        x = self.avgpool(x) ## (bs, num_filters * 4, 1)
        x = self.flatten(x) ## (bs, num_filters * 4)
        out = self.linear(x) ## (bs, num_classes)
        return out
                
if __name__ == "__main__":
    sample = torch.rand(3, 85, 120)
    num_classes = 31
    _, in_channels, seq_length = sample.size()
    inception_block = InceptionBlock(in_channels = in_channels)
    inception_res_block = Inception_ResidualBlock(in_channels = in_channels)
    inception_time = InceptionTime(in_channels=in_channels, sequence_len=seq_length, num_classes=num_classes)
    out = inception_block(sample)
    print("Inception Block Output Size:",out.size())
    out = inception_res_block(sample)
    print("Inception Residual Block Output Size:",out.size())
    out = inception_time(sample)
    print("Inception Time Output Size:", out.size())