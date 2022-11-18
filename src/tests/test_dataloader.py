import config
import torch

from dev.dataloader import SpeechDataLoader

def test_loading_pickle():
    batch_size = 3
    padding_length = 20000
    fft_window, fft_overlap = 512, 128
    sequence_len = (padding_length - fft_overlap) // fft_overlap + 2
    mel_channels = 128
    dataloader =  SpeechDataLoader(data_dir = config.DATA_PATH,
                                   batch_size = batch_size,
                                   sample_rate = config.SAMPLE_RATE,
                                   padding_length = padding_length,
                                   fft_window = fft_window,
                                   fft_overlap = fft_overlap,
                                   mel_channels = mel_channels,
                                   load_pickle = True,
                                   )
    dataloader.setup()
    train_loader = dataloader.train_loader()
    val_loader = dataloader.validation_loader()
    test_loader = dataloader.test_loader()
    predict_loader = dataloader.predict_loader()

    inputs, targets = next(iter(train_loader))
    assert inputs.size() == torch.Size((batch_size, mel_channels, sequence_len)), "Pickle Training Dataset Images dimension mismatches"
    assert len(targets) == batch_size, "Pickle Training Dataset labels dimension mismatches"
    assert isinstance(targets[0].item(), int), "Incorrect Target Data Types"
    assert min(targets) >= 0 and max(targets) <= len(config.LABELS) - 1, "Incorrect Target Values" 
    
    inputs, targets = next(iter(val_loader))
    assert inputs.size() == torch.Size((batch_size, mel_channels, sequence_len)), "Pickle Validation Dataset Images dimension mismatches"
    assert len(targets) == batch_size, "Pickle Validation Dataset labels dimension mismatches"
    assert isinstance(targets[0].item(), int), "Incorrect Target Data Types"
    assert min(targets) >= 0 and max(targets) <= len(config.LABELS) - 1, "Incorrect Target Values" 
    
    inputs, _ = next(iter(test_loader))
    assert inputs.size() == torch.Size((batch_size, mel_channels, sequence_len)), "Pickle Testing Dataset Images dimension mismatches"
    assert len(targets) == batch_size, "Pickle Test Dataset labels dimension mismatches"
    assert isinstance(targets[0].item(), int), "Incorrect Target Data Types"
    assert min(targets) >= 0 and max(targets) <= len(config.LABELS) - 1, "Incorrect Target Values" 
    
    inputs = next(iter(predict_loader))
    assert inputs.size() == torch.Size((batch_size, mel_channels, sequence_len)), "Pickle Prediction Dataset Images dimension mismatches"    
    
# def test_loading_wav():
#     batch_size = 3
#     padding_length = 20000
#     fft_window, fft_overlap = 512, 128
#     sequence_len = (padding_length - fft_overlap) // fft_overlap + 2
#     mel_channels = 128
#     dataloader =  SpeechDataLoader(data_dir = config.DATA_PATH,
#                                    batch_size = batch_size,
#                                    sample_rate = config.SAMPLE_RATE,
#                                    padding_length = padding_length,
#                                    fft_window = fft_window,
#                                    fft_overlap = fft_overlap,
#                                    mel_channels = mel_channels,
#                                    load_pickle = False,
#                                    )
    
#     dataloader.setup()
#     train_loader = dataloader.train_loader()
#     val_loader = dataloader.validation_loader()
#     test_loader = dataloader.test_loader()
#     predict_loader = dataloader.predict_loader()

#     inputs, targets = next(iter(train_loader))
#     assert inputs.size() == torch.Size((batch_size, mel_channels, sequence_len)), "Wav Training Dataset Images dimension mismatches"
#     assert len(targets) == batch_size, "Wav Training Dataset labels dimension mismatches"
#     assert isinstance(targets[0].item(), int), "Incorrect Target Data Types"
#     assert min(targets) >= 0 and max(targets) <= len(config.LABELS) - 1, "Incorrect Target Values" 
    
#     inputs, targets = next(iter(val_loader))
#     assert inputs.size() == torch.Size((batch_size, mel_channels, sequence_len)), "Wav Validation Dataset Images dimension mismatches"
#     assert len(targets) == batch_size, "Wav Validation Dataset labels dimension mismatches"
#     assert isinstance(targets[0].item(), int), "Incorrect Target Data Types"
#     assert min(targets) >= 0 and max(targets) <= len(config.LABELS) - 1, "Incorrect Target Values" 
    
#     inputs, _ = next(iter(test_loader))
#     assert inputs.size() == torch.Size((batch_size, mel_channels, sequence_len)), "Wav Testing Dataset Images dimension mismatches"
#     assert len(targets) == batch_size, "Wav Test Dataset labels dimension mismatches"
#     assert isinstance(targets[0].item(), int), "Incorrect Target Data Types"
#     assert min(targets) >= 0 and max(targets) <= len(config.LABELS) - 1, "Incorrect Target Values" 
    
#     inputs = next(iter(predict_loader))
#     assert inputs.size() == torch.Size((batch_size, mel_channels, sequence_len)), "Wav Prediction Dataset Images dimension mismatches"