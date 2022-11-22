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
* Inception Time
* U-Time

#### Sequence Translation with CTCLoss
* LSTM/RNN/GRU

## Results

|**Model**|**Macro Avg Accuracy (Val)**|**Macro Avg Accuracy (Test)**|
| :-------------: | :-----------------------: | :---------------: |
|RNN (Classification)|To be updated|To be updated|
|InceptionTime (Classification)|89.28%|88.07%|
|UTime (Classification)|To be updated|To be updated|
|LSTM (CTCLoss)|To be updated|To be updated|

## To do List
1. Update Multi-Class micro-average accuracy instead of macro-average accuracy
2. Kaggle inference: Only 12 classes (Convert some claasses to Unknown)
3. MODELS

   **Training Using CTCLoss**
* How to setup output character dictionary
* How to infer? Greedy Decode? Beam Search?
* In cases where inference doesnt match, how to map close words to corresponding labels.

  **Other RNN Models**
* RNN
* Utime

4. Create API & Use API to infer Predict dataset