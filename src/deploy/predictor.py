"""For Deployment
"""
# from typing import Union, Sequence, Callable
# import config
# import librosa
# import numpy as np
# import torch
# from dev.utils import load_model, convert_kaggle_label, ff_transform
# from pathlib import Path

# class ClassificationPredictor():
#     def __init__(
#         self,
#         model: Union[Callable, str],

#     ):
#         if isinstance(model, str):
#             self.model_path = model
#             self.model_type = self.model_path.split("/")[-2]
#             self.model = config.MODEL_PARAMS["self.model_type"]["instance"]()
#             load_model(self.model, self.model_path)

#         else:
#             self.model = model
#             self.model_type = str(model)

#     def predict(self, audio: Union[np.array, str]):
#         if isinstance(audio, str):
#             audio = librosa.load(audio, sr = config.SAMPLE_RATE)

#         self.model.to(config.DEVICE)
#         self.model.eval()
#         with torch.no_grad():
#             return

#     def batch_predict(self, ):
#         self.model.eval()


#     def predict_kaggle(self, )


# def inference(audio: Union[str, Path, Sequence], model: Callable, decoder: ):
#     if isinstance(input, str) or isinstance(input, Path):
#         audio, _ = librosa.load(audio, sr = config.SAMPLE_RATE)

#     if len(audio) < config.SEQUENCE_LEN:
#         audio = np.pad(audio, (0, config.SEQUENCE_LEN-len(audio)), "linear_ramp")
#     else:
#         audio = audio[:config.SEQUENCE_LEN]

#     output = model(input)
#     label = config.ID2LABEL[output]
#     return label
