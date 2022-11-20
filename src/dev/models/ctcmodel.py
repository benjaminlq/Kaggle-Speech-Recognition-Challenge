"""Translation Model using Connectionist Temporal Classification (CTC)
"""
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
        """Connectionist Temporal Classification Model

        Args:
            vocab_size (int): Vocab Size (Output)
            channels_no (int): Number of input channels for each timestep in sequence.
            input_size (int, optional): No of input channels to be input into RNN/LSTM/GRU. Defaults to 64.
            hidden_size (int, optional): Hidden Size. Defaults to 32.
            dropout (float, optional): Dropout Rate. Defaults to 0.25.
            bidirectional (bool, optional): Bidirectional RNN/LSTM/GRU. Defaults to True.
        """
        super(CTCModel, self).__init__()
        ## Parameters
        self.vocab_size = vocab_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.channels_no = channels_no
        self.bidirectional = bidirectional

        ## Blocks
        self.linear = nn.Linear(self.channels_no, self.input_size)
        self.rnn = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=2,
            bidirectional=self.bidirectional,
            dropout=dropout,
            batch_first=True,
        )

        if self.bidirectional:
            self.output = nn.Linear(self.hidden_size * 2, self.vocab_size)
        else:
            self.output = nn.Linear(self.hidden_size, self.vocab_size)

        self.logsoftmax = nn.LogSoftmax(dim=2)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, inputs: torch.tensor) -> torch.tensor:
        """Forward Pass

        Args:
            inputs (torch.tensor): Inputs. Dimension = (bs, no_channels, seq_len)

        Returns:
            torch.tensor: Logsoftmax of probability of each character at each timestep. Dimension = (bs, seq_len, vocab_size)
        """
        ## Inputs Dimension = (bs, no_channels, seq_len)
        x = inputs.permute(0, 2, 1)  # (bs, seq_len, no_channels)
        x = F.relu(self.linear(x))
        x = self.dropout(x)
        x, _ = self.rnn(x)
        x = self.output(x)  # (bs, vocab_size)
        out = self.logsoftmax(x)
        return out

    def __str__(self):
        """
        Returns:
            str: Model Type Name
        """
        return "ctcmodel"


if __name__ == "__main__":
    batch_size = 4
    vocab_size = 26
    channels_no = 128
    sequence_len = 126
    sample = torch.rand(batch_size, channels_no, sequence_len)
    model = CTCModel(vocab_size=vocab_size, channels_no=channels_no)

    output = model(sample)
    print(output.size())
