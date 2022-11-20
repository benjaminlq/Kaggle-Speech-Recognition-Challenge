"""Speech DataLoader
"""
import os
import pickle
from pathlib import Path
from typing import Callable, Literal, Optional, Union

import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms

import config
from config import LOGGER
from dev.dataset import SpeechDataset


class SpeechDataLoader(DataLoader):
    def __init__(
        self,
        data_dir: Union[str, Path] = config.DATA_PATH,
        transform: Optional[Callable] = None,
        batch_size: int = 32,
        fft_type: Literal["spectrogram", "mfcc"] = "mfcc",
        sample_rate: int = config.SAMPLE_RATE,
        padding_type: str = "linear_ramp",
        padding_length: int = config.PADDING_LENGTH,
        fft_window: int = config.FFT_WINDOW,
        fft_overlap: int = config.FFT_OVERLAP,
        mel_channels: int = config.MEL_CHANNELS,
        load_pickle: bool = True,
    ):
        """Speech DataLoader.

        Args:
            data_dir (Union[str, Path], optional): Path to data file or data folder. Defaults to config.DATA_PATH.
            transform (Optional[Callable], optional): Additional Transform & Augmentation. Defaults to None.
            batch_size (int, optional): Batch Size. Defaults to 32.
            fft_type (Literal["spectrogram";, "mfcc"], optional): Method to use for Fourier Transform to generate feature for each timestamp. If "spectrogram",
            use Discrete Short Time Fourier Transform to convert time domain to frequency domain with feature size = FFT_WINDOW // 2 + 1. If "mfcc", use Mel Frequency
            cepstral coefficient to generate features of size = MEL_CHANNELS. Defaults to "mfcc".
            sample_rate (int, optional): sample rate of sound files. Defaults to config.SAMPLE_RATE.
            padding_type (str, optional): type of padding. Defaults to "linear_ramp".
            padding_length (int, optional): length of padded sequence. Defaults to config.PADDING_LENGTH.
            fft_window (int, optional): window for FFT. Defaults to config.FFT_WINDOW.
            fft_overlap (int, optional): window overlapping size for FFT. Defaults to config.FFT_OVERLAP.
            mel_channels (int, optional): No of MEL channels if "mfcc" is used for fft_type. Defaults to config.MEL_CHANNELS.
            load_pickle (bool, optional): If True, load data from pickle files. If False, DataLoader contains list of file paths to be read. Defaults to True.
        """
        self.labels = config.LABELS
        self.label2id = config.LABEL2ID
        self.id2label = config.ID2LABEL

        self.num_classes = len(self.labels)
        self.load_pickle = load_pickle
        self.batch_size = batch_size

        if isinstance(data_dir, Path):
            self.data_dir = str(data_dir)
        else:
            self.data_dir = data_dir

        if self.load_pickle:
            ### Load Pre-Loaded full dataset
            self.data_dir = os.path.join(data_dir, "pickle")
            LOGGER.info(f"Loading Pickle data files from {self.data_dir}")

            ### Load pickle data ###
            with open(os.path.join(data_dir, "pickle", "train.pkl"), "rb") as f:
                self.train_images, self.train_targets = pickle.load(f)
            with open(os.path.join(data_dir, "pickle", "validation.pkl"), "rb") as f:
                self.val_images, self.val_targets = pickle.load(f)
            with open(os.path.join(data_dir, "pickle", "test.pkl"), "rb") as f:
                self.test_images, self.test_targets = pickle.load(f)
            with open(os.path.join(data_dir, "pickle", "predict.pkl"), "rb") as f:
                self.predict_images = pickle.load(f)
            LOGGER.info("Finished Loading Pickle Data Files")

            ### Convert labels to ids:
            self.train_targets = [self.label2id[label] for label in self.train_targets]
            self.val_targets = [self.label2id[label] for label in self.val_targets]
            self.test_targets = [self.label2id[label] for label in self.test_targets]

        else:
            ### Prepare paths to each wav files. Need to test for code runability.
            with open(
                os.path.join(self.data_dir, "train", "validation_list.txt"), "r"
            ) as f:
                self.val_images = [file.rstrip() for file in f.readlines()]
                self.val_targets = [
                    self.label2id[path.split("/")[0]] for path in self.val_images
                ]
                self.val_images = [
                    os.path.join(data_dir, "train", "audio", path)
                    for path in self.val_images
                ]
            with open(
                os.path.join(self.data_dir, "train", "testing_list.txt"), "r"
            ) as f:
                self.test_images = [file.rstrip() for file in f.readlines()]
                self.test_targets = [
                    self.label2id[path.split("/")[0]] for path in self.test_images
                ]
                self.test_images = [
                    os.path.join(data_dir, "train", "audio", path)
                    for path in self.test_images
                ]

            self.train_images = []
            self.train_targets = []

            for label in self.labels[1:]:
                filelist = os.listdir(
                    os.path.join(self.data_dir, "train", "audio", label)
                )
                for file in filelist:
                    filepath = label + "/" + file
                    if (
                        filepath not in self.val_images
                        and filepath not in self.test_images
                    ):
                        self.train_images.append(
                            os.path.join(self.data_dir, "train", "audio", filepath)
                        )
                        self.train_targets.append(self.label2id[label])

            ## Setup Silence
            silence_list = os.listdir(
                os.path.join(self.data_dir, "train", "audio", "silence")
            )
            np.random.shuffle(silence_list)
            silence_train_images = [
                os.path.join(self.data_dir, "train", "audio", "silence", silence_file)
                for silence_file in silence_list[:2300]
            ]
            self.train_images += silence_train_images
            self.train_targets += ["silence"] * len(silence_train_images)

            silence_val_images = [
                os.path.join(self.data_dir, "train", "audio", "silence", silence_file)
                for silence_file in silence_list[2300:2550]
            ]
            self.val_images += silence_val_images
            self.val_targets += ["silence"] * len(silence_val_images)

            silence_test_images = [
                os.path.join(self.data_dir, "train", "audio", "silence", silence_file)
                for silence_file in silence_list[2550:]
            ]
            self.test_images += silence_test_images
            self.test_targets += ["silence"] * len(silence_test_images)

            self.predict_images = [
                os.path.join(data_dir, "test", "audio", path)
                for path in os.listdir(os.path.join(data_dir, "test", "audio"))
            ]

        ## FFT parameters
        self.sample_rate = sample_rate
        self.padding_type = padding_type
        self.padding_length = padding_length
        self.fft_type = fft_type
        self.fft_window = fft_window
        self.fft_overlap = fft_overlap
        self.mel_channels = mel_channels

        ## Augmentation

        if transform:
            self.transforms = transform
        else:
            self.transforms = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(config.MEAN, config.STD),
                ]
            )

    def setup(self):
        """Setup Train, Validation, Test & Predict DataLoader"""
        self.train_dataset = SpeechDataset(
            data_images=self.train_images,
            load_pickle=self.load_pickle,
            targets=self.train_targets,
            transform=self.transforms,
            fft_type=self.fft_type,
            sample_rate=self.sample_rate,
            padding_type=self.padding_type,
            padding_length=self.padding_length,
            fft_window=self.fft_window,
            fft_overlap=self.fft_overlap,
            mel_channels=self.mel_channels,
            augmentation=True,
        )

        self.val_dataset = SpeechDataset(
            data_images=self.val_images,
            load_pickle=self.load_pickle,
            targets=self.val_targets,
            transform=self.transforms,
            fft_type=self.fft_type,
            sample_rate=self.sample_rate,
            padding_type=self.padding_type,
            padding_length=self.padding_length,
            fft_window=self.fft_window,
            fft_overlap=self.fft_overlap,
            mel_channels=self.mel_channels,
            augmentation=False,
        )

        self.test_dataset = SpeechDataset(
            data_images=self.test_images,
            load_pickle=self.load_pickle,
            targets=self.test_targets,
            transform=self.transforms,
            fft_type=self.fft_type,
            sample_rate=self.sample_rate,
            padding_type=self.padding_type,
            padding_length=self.padding_length,
            fft_window=self.fft_window,
            fft_overlap=self.fft_overlap,
            mel_channels=self.mel_channels,
            augmentation=False,
        )

        self.inference_dataset = SpeechDataset(
            data_images=self.predict_images,
            load_pickle=self.load_pickle,
            transform=self.transforms,
            fft_type=self.fft_type,
            sample_rate=self.sample_rate,
            padding_type=self.padding_type,
            padding_length=self.padding_length,
            fft_window=self.fft_window,
            fft_overlap=self.fft_overlap,
            mel_channels=self.mel_channels,
            augmentation=False,
        )

    def train_loader(self) -> DataLoader:
        """Train DataLoader

        Returns:
            DataLoader: Train DataLoader
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=config.NUM_WORKERS,
        )

    def validation_loader(self) -> DataLoader:
        """Validation DataLoader

        Returns:
            DataLoader: Validation DataLoader
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=config.NUM_WORKERS,
        )

    def test_loader(self) -> DataLoader:
        """Test DataLoader

        Returns:
            DataLoader: Test DataLoader
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=config.NUM_WORKERS,
        )

    def predict_loader(self) -> DataLoader:
        """Predict DataLoader

        Returns:
            DataLoader: Predict DataLoader
        """
        return DataLoader(
            self.inference_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=config.NUM_WORKERS,
        )


if __name__ == "__main__":
    dataloader = SpeechDataLoader()
    dataloader.setup()
    train_loader = dataloader.train_loader()
    val_loader = dataloader.validation_loader()
    test_loader = dataloader.test_loader()
    predict_loader = dataloader.predict_loader()

    inputs, targets = next(iter(train_loader))
    print(inputs.size(), len(targets))
