"""Basic RNN model
"""
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """Self Attention Module"""

    def __init__(
        self,
        hidden_size: int,
        proj_hidden_size: int,
    ):
        """Self Attention Module

        Args:
            hidden_size (int): Hidden Size of Encoder Outputs
            proj_hidden_size (int): Hidden Size of Self Attention hidden layer
        """
        super(Attention, self).__init__()
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, proj_hidden_size),
            nn.ReLU(True),
            nn.Linear(proj_hidden_size, 1),
        )

    def forward(self, encoder_outputs: torch.tensor) -> torch.tensor:
        """Forward Propagation

        Args:
            encoder_outputs (torch.tensor): Output hidden states for all encoder timestamps. Dimension: bs, sequence_length, hidden_size

        Returns:
            torch.tensor: Weighted tensor for all encoder hidden states
        """
        att_energy = self.projection(encoder_outputs)  # (bs, seq_len, 1)
        att_weights = F.softmax(att_energy.squeeze(2), dim=1)  # (bs, seq_len)
        outputs = (encoder_outputs * att_weights.unsqueeze(2)).sum(
            dim=1
        )  # (bs, hidden_size)
        return outputs


class RNNModel(nn.Module):
    """RNN Model"""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        sequence_len: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        rnn_type: Literal["RNN", "LSTM", "GRU"] = "GRU",
        bidirectional: bool = True,
        dropout: float = 0.0,
        use_attention: bool = True,
        proj_size: int = 64,
    ):
        """RNN Model

        Args:
            in_channels (int): Number of input channels for each timestep.
            num_classes (int): Sequence Length
            sequence_len (int): Number of output class labels
            hidden_size (int, optional): RNN hidden size to pass to next time step. Defaults to 64.
            num_layers (int, optional): Number of RNN layers. Defaults to 2.
            rnn_type (str, "LSTM", "RNN" or "GRU"): Type of RNN architecture. Defaults to "LSTM". Need to be UPPERCASE.
            bidirectional (bool, optional): Use bidirectional RNN. Defaults to True.
            dropout (float, optional): Dropout Rate. Defaults to 0.25.
            use_attention (bool, optional): Use self-attention. Defaults to False.
            proj_size (int, optional): Hidden size for self-attention network. Default to 64.
        """

        super(RNNModel, self).__init__()

        self.bidirectional = bidirectional
        self.sequence_len = sequence_len
        self.use_attention = use_attention

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

        self.dropout = nn.Dropout(p=dropout)

        if use_attention:
            if self.bidirectional:
                self.linear = nn.Linear(4 * hidden_size, num_classes)
                self.self_attention = Attention(2 * hidden_size, proj_size)
            else:
                self.linear = nn.Linear(2 * hidden_size, num_classes)
                self.self_attention = Attention(hidden_size, proj_size)
        else:
            self.self_attention = None
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
        x = inputs.permute(0, 2, 1)  # Dimension = (bs, sequence_len, input_size)
        # x = self.dropout(inputs)
        outs, _ = self.rnn(x)  # (bs, sequence_len, hidden_size)
        rnn_out = outs[:, -1, :]  # (bs, hidden_size)
        att_out = self.self_attention(outs)  # (bs, hidden_size)
        final_out = torch.cat((rnn_out, att_out), dim=1)

        probs = self.linear(final_out)  # (bs, num_classes)

        return probs

    def __str__(self):
        """Model Name"""
        if self.use_attention:
            return "RNN_ATT"
        else:
            return "RNN"


if __name__ == "__main__":
    model = RNNModel(
        in_channels=128, num_classes=31, sequence_len=126, use_attention=True
    )
    sample_batch = torch.rand(5, 128, 126)
    out = model(sample_batch)
    print(out.size())
