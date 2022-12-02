"""For Deployment
"""
import os
import pickle
from typing import Callable, Literal, Optional, Union

import librosa
import numpy as np
import torch

import config
from dev.inference import BeamSearchCTCDecoder, GreedyCTCDecoder
from dev.utils import convert_kaggle_label, ff_transform, load_model


class Predictor:
    """Predictor Interface"""

    def __init__(
        self,
        model: Union[Callable, str],
    ):
        """Predictor Interface Module

        Args:
            model (Union[Callable, str]): Model Instance or Path to Model Checkpoint
        """
        if isinstance(model, str):
            self.model_path = model
            self.model_name = self.model_path.split("/")[-2]
            self.model = config.MODEL_PARAMS["self.model_name"]["instance"]()
            load_model(self.model, self.model_path)

        else:
            self.model = model
            self.model_name = str(model)

        self.model.eval()
        self.model.to(config.DEVICE)
        self.model_type = config.MODEL_PARAMS["self.model_name"]["type"]

    def predict(
        self, audio: Union[np.array, str], filename: Optional[str] = None
    ) -> NotImplementedError:
        """Class function for single sample prediction

        Args:
            audio (Union[np.array, str]): If type is np.array, audio is sequence of 1D array of sound. If str type, audio is path to soundfile.
            filename (Optional[str], optional): File Name corresponding to the input. Defaults to None.

        Returns:
            NotImplementedError: Inference Prediction on the sample input
        """
        raise NotImplementedError

    def batch_predict(self, audios: Union[list, str], filenames: Optional[list] = None):
        """Batch Prediction

        Args:
            audios (Union[list, str]): If List, inputs are list of np.array or str-type paths to batch of audio files. If str, provide path to pickle data files.
            filenames (list, optional): List of filenames to provide. Defaults to None.

        Returns:
            _type_: _description_
        """
        if isinstance(audios, str):
            with open(audios, "rb") as f:
                audios, filenames = pickle.load(f)
        elif isinstance(audios[0], str):
            filenames = [audio.split("/")[-1] for audio in audios]

        if filenames:
            assert len(filenames) == len(
                audios
            ), "Length of filename list and audio list mismatch."

        predictions = (
            [self.predict(audio, filenames[idx]) for idx, audio in enumerate(audios)]
            if filenames
            else [self.predict(audio) for audio in audios]
        )

        return predictions

    def predict_kaggle(
        self,
        audios: Union[list, str],
        outfolder: str = str(config.MODEL_PATH / "kaggle"),
    ):
        """Convert Model Raw Prediction to Kaggle Template and write to .txt file.

        Args:
            audios (Union[list, str]): If List, inputs are list str-type paths to batch of audio files. If str, provide path to pickle data files.
            outfolder (str, optional): Path to write output artifact.
        """
        predictions = self.batch_predict(audios)
        kaggle_predictions = [
            (filename, convert_kaggle_label(pred)) for (filename, pred) in predictions
        ]
        if not os.path.exists(outfolder):
            os.makedirs(outfolder, exist_ok=True)
        with open(os.path.join(outfolder, "kaggle_predictions.txt"), "w") as f:
            f.write("fname,label\n")
            for filename, kaggle_pred in kaggle_predictions:
                f.write(f"{filename},{kaggle_pred}\n")
        print("Inference Completed!!!")


class ClassificationPredictor(Predictor):
    """Predictor Module using Classification Models"""

    def __init__(
        self,
        model: Union[Callable, str],
    ):
        """Predictor Module using Classification Models

        Args:
            model (Union[Callable, str]): Model Instance or Path to Model Checkpoint
        """
        super(ClassificationPredictor, self).__init__(model)

    def predict(self, audio: Union[np.array, str], filename: Optional[str] = None):
        """Class function for single sample prediction

        Args:
            audio (Union[np.array, str]): If type is np.array, audio is sequence of 1D array of sound. If str type, audio is path to soundfile.
            filename (Optional[str], optional): File Name corresponding to the input. Defaults to None.

        Returns:
            (str, str) or str: (filename, prediction) or (prediction)
        """
        if isinstance(audio, str):
            filename = audio.split("/")[-1]
            audio = librosa.load(audio, sr=config.SAMPLE_RATE)

        if len(audio) < config.PADDING_LENGTH:
            audio = np.pad(
                audio, (0, config.PADDING_LENGTH - len(audio)), "linear_ramp"
            )

        audio = audio[: self.padding_length]
        S = ff_transform(
            audio,
            config.FFT_WINDOW,
            config.FFT_OVERLAP,
            config.SAMPLE_RATE,
            config.MEL_CHANNELS,
        )

        input = torch.tensor(S, dtype=torch.float32).unsqueeze(0).to(config.DEVICE)

        with torch.no_grad():
            out = self.model(input)
            pred = torch.argmax(out).squeeze(0).detach().cpu().item()

        return (filename, config.ID2LABEL[pred]) if filename else config.ID2LABEL[pred]


class CTCPredictor(Predictor):
    """Predictor Module using CTC Models"""

    def __init__(
        self,
        model: Union[Callable, str],
        decoder: Literal["greedy", "beam"] = "greedy",
    ):
        """Predictor Module using CTC Models

        Args:
            model (Union[Callable, str]): Model Instance or Path to Model Checkpoint
        """
        super(ClassificationPredictor, self).__init__(model)
        self.decoder = (
            GreedyCTCDecoder() if decoder == "greedy" else BeamSearchCTCDecoder()
        )

    def predict(self, audio: Union[np.array, str], filename: Optional[str] = None):
        """Class function for single sample prediction

        Args:
            audio (Union[np.array, str]): If type is np.array, audio is sequence of 1D array of sound. If str type, audio is path to soundfile.
            filename (Optional[str], optional): File Name corresponding to the input. Defaults to None.

        Returns:
            (str, str) or str: (filename, prediction) or (prediction)
        """
        if isinstance(audio, str):
            filename = audio.split("/")[-1]
            audio = librosa.load(audio, sr=config.SAMPLE_RATE)

        if len(audio) < config.PADDING_LENGTH:
            audio = np.pad(
                audio, (0, config.PADDING_LENGTH - len(audio)), "linear_ramp"
            )

        audio = audio[: self.padding_length]
        S = ff_transform(
            audio,
            config.FFT_WINDOW,
            config.FFT_OVERLAP,
            config.SAMPLE_RATE,
            config.MEL_CHANNELS,
        )

        input = torch.tensor(S, dtype=torch.float32).unsqueeze(0).to(config.DEVICE)

        with torch.no_grad():
            log_probs = self.model(input)
            _, _, preds = self.decoder.decode(log_probs)
        pred = preds[0]

        return (filename, config.ID2LABEL[pred]) if filename else config.ID2LABEL[pred]
