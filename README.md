# TensorFlow-Speech-Recognition-Challenge

## DataSet
Speech Recognition Dataset from Kaggle Tensorflow Speech Recognition Challenge

https://www.kaggle.com/competitions/tensorflow-speech-recognition-challenge

### Pickle Dataset
https://www.kaggle.com/datasets/lequan2902/speech-recognition-16000sr
* Sample Rate = 16000
* Silence files split into small files with duration 1 sec (np.array of length 16000 (sr = 16000))
* 3 files: Train, Val, Test with data & labels
* Predict Set contains samples without labels.

## Installation
### Virtual Env Creation & Activation

* `python3 -m venv venv` for initialising the virtual environment
* `source venv/bin/activate` for activating the virtual environment

### Dependency Installation

The following commands shall be ran **after activating the virtual environment**.

* `pip install --upgrade pip` for upgrading the pip
* `pip install -r requirements.txt` for the functional dependencies
* `pip install -r requirements-dev.txt` for the development dependencies. (should include `pre-commit` module)
* `pre-commit install` for installing the precommit hook
* `pip install -e .` for the files/modules in `src` to be accessed as a package.

## Methodology

### Model Architecture
#### Multiclass Classification
* RNN
* RNN With Self-Attention
* Inception Time

#### Sequence Translation with CTCLoss
* LSTM/RNN/GRU
* Use CTCLoss to classify tokens of characters, including a blank ("-") token and a silence token ("|")
* Use GreedyDecoder and BeamSearchDecoder for converting sequence of tokens to character sequence
* If sequence does not match provided label, use edit_distance to find the nearest candidate label.

## Results

|**Model**|**Macro Avg Accuracy (Val)**|**Macro Avg Accuracy (Test)**|
| :-------------: | :-----------------------: | :---------------: |
|LSTM (Classification)|68.30%|67.32%|
|GRU with Self-Attention (Classification)|80.00%|78.54%|
|InceptionTime (Classification)|89.28%|88.07%|
|GRU (CTCLoss)|To be updated|To be updated|

* Stop Training LSTM (Classification) model after 75 training epochs due to low computational resources. Model Val Loss is still improving at slow rate.
* Other Models performance reported after 30 training epochs.
