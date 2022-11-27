"""Basic RNN model
"""
from typing import Literal

import torch
import torch.nn as nn


class RNNModel(nn.Module):
    """RNN Model

    Args:
        nn (_type_): _description_
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        sequence_len: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        rnn_type: Literal["RNN", "LSTM", "GRU"] = "LSTM",
        bidirectional: bool = True,
        dropout: float = 0.0,
        self_attention: bool = False,
    ):
        """_summary_

        Args:
            in_channels (int): Number of input channels for each timestep.
            num_classes (int): Sequence Length
            sequence_len (int): Number of output class labels
            hidden_size (int, optional): RNN hidden size to pass to next time step. Defaults to 64.
            num_layers (int, optional): Number of RNN layers. Defaults to 2.
            rnn_type (str, "LSTM", "RNN" or "GRU"): Type of RNN architecture. Defaults to "LSTM". Need to be UPPERCASE.
            bidirectional (bool, optional): Use bidirectional RNN. Defaults to True.
            dropout (float, optional): Dropout Rate. Defaults to 0.25.
            self_attention (bool, optional): Use self-attention. Defaults to False.
        """

        super(RNNModel, self).__init__()

        self.bidirectional = bidirectional
        self.sequence_len = sequence_len
        self.self_attention = self_attention
        self.rnn_type = rnn_type
        rnn_obj = getattr(nn, rnn_type)
        self.rnn = rnn_obj(
            input_size=in_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=False,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )

        # if rnn_type == "rnn":
        #     self.rnn = nn.RNN(
        #         input_size=in_channels,
        #         hidden_size=hidden_size,
        #         num_layers=num_layers,
        #         bias=False,
        #         batch_first=True,
        #         dropout=dropout,
        #         bidirectional=bidirectional,
        #     )
        # elif rnn_type == "lstm":
        #     self.rnn = nn.LSTM(
        #         input_size=in_channels,
        #         hidden_size=hidden_size,
        #         num_layers=num_layers,
        #         bias=False,
        #         batch_first=True,
        #         dropout=dropout,
        #         bidirectional=bidirectional,
        #     )
        # else:
        #     self.rnn = nn.GRU(
        #         input_size=in_channels,
        #         hidden_size=hidden_size,
        #         num_layers=num_layers,
        #         bias=False,
        #         batch_first=True,
        #         dropout=dropout,
        #         bidirectional=bidirectional,
        #     )

        self.dropout = nn.Dropout(p=dropout)
        if self.bidirectional:
            self.linear = nn.Linear(2 * hidden_size, num_classes)
        else:
            self.linear = nn.Linear(hidden_size, num_classes)

    def forward(self, inputs: torch.tensor) -> torch.tensor:
        """Forward Pass

        Args:
            inputs (torch.tensor): Input Mini Batch

        Returns:
            torch.tensor: Output Probability
        """
        ## Input Dimension = (bs, input_size, sequence_len)
        inputs = inputs.permute(0, 2, 1)  # Dimension = (bs, sequence_len, input_size)
        x = self.dropout(inputs)
        outs, _ = self.rnn(x)
        probs = self.linear(outs[:, -1, :])  # Dimension = (bs, num_classes)
        return probs

    def __str__(self):
        """Model Name"""
        if self.self_attention:
            return "RNN_ATT"
        else:
            return "RNN"

    # def attention(self, inputs):


if __name__ == "__main__":
    model = RNNModel(in_channels=128, num_classes=31, sequence_len=126)
    sample_batch = torch.rand(5, 128, 126)
    out = model(sample_batch)
    print(out.size())
