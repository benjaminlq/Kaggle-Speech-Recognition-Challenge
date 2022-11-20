"""Unit Tests for models
"""
import torch

from dev.models.ctcmodel import CTCModel
from dev.models.inceptiontime import InceptionTime

models = {}


def test_classification():
    """Test Classification Models"""
    in_channels, sequence_len = 128, 126
    num_classes = 31
    batch_size = 3
    inception_model = InceptionTime(
        in_channels=in_channels, sequence_len=sequence_len, num_classes=num_classes
    )
    sample_batch = torch.rand(batch_size, in_channels, sequence_len)
    output = inception_model(sample_batch)

    assert output.size() == torch.Size(
        (batch_size, num_classes)
    ), "Output Size Mismatch"


def test_translation():
    """Test Translation Models"""
    in_channels, sequence_len = 128, 126
    vocab_size = 26
    batch_size = 3
    ctc_model = CTCModel(vocab_size, channels_no=in_channels)
    sample_batch = torch.rand(batch_size, in_channels, sequence_len)
    output = ctc_model(sample_batch)

    assert output.size() == torch.Size((batch_size, sequence_len, vocab_size))
