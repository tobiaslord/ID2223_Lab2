# Lab 2

Scalabale Machine Learning (ID2223)

## Introduction

In September 2022 authors Alec Radford et al. released Whisper, a pre-trained model for speech recognition (ASR). Whisper is trained on labeled audo-transcription data. Since a big part of the data its been trained on is multilingual ASR data, results on the data can be used in many langugages.

## Purpose

The purpose of this lab is to demonstrate the capabilities of Whisper using the mother tongue of the authors, in this case Swedish. First, we will follow the given sample code [https://colab.research.google.com/github/sanchit-gandhi/notebooks/blob/main/fine_tune_whisper.ipynb] and then make some additions to it.

- Refactor the code into
  - Feature engineering pipeline which can leverage a CPU
  - Training pipeline which makes use of an GPU
  - Inference program to display the model's capabilities
- Ponder and possibly show how we can improve results using
  - a model-centric approach (e.g., tune hyperparameters, change the fine-tuning model architecture, etc)
  - a data-centric approach - identify new data sources that enable you to train a better model that one provided in the blog post

### Links

#### Inference

### Data-centric approach

- Make use of accent, age, gender, locale, etc. to make the model fit the end user better.

## Implementation

### Data


### Model