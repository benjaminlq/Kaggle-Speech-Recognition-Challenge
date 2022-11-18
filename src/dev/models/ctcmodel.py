import torch
import torch.nn as nn
import torch.nn.functional as F

class CTCModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        channels_no: int,
        input_size: int = 64,
        hidden_size: int = 32,
        dropout: float = 0.25,
        bidirectional: bool = True,
    ):
        super(CTCModel, self).__init__()
        ## Parameters
        self.vocab_size = vocab_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.channels_no = channels_no
        self.bidirectional = bidirectional
        
        ## Blocks
        self.linear = nn.Linear(self.channels_no, self.input_size)
        self.rnn = nn.LSTM(input_size = self.input_size, hidden_size = self.hidden_size,
                           num_layers = 2, bidirectional = self.bidirectional, dropout = dropout, batch_first = True)
        
        if self.bidirectional:
            self.output = nn.Linear(self.hidden_size * 2, self.vocab_size)
        else:
            self.output = nn.Linear(self.hidden_size, self.vocab_size)
        
        self.logsoftmax = nn.LogSoftmax(dim = 2)
        self.dropout = nn.Dropout(p = dropout)
        
    def forward(self, inputs):
        ## Inputs Dimension = (bs, no_channels, seq_len)
        x = inputs.permute(0,2,1) # (bs, seq_len, no_channels)
        x = F.relu(self.linear(x)) 
        x = self.dropout(x)
        x, _ = self.rnn(x)
        x = self.output(x) # (bs, vocab_size)
        out = self.logsoftmax(x) 
        return out
    
    def __str__(self):
        return "ctcmodel"
    
if __name__ == "__main__":
    batch_size = 4
    vocab_size = 26
    channels_no = 128
    sequence_len = 126
    sample = torch.rand(batch_size, channels_no, sequence_len)
    model = CTCModel(vocab_size = vocab_size, channels_no = channels_no)
    
    output = model(sample)
    print(output.size())    