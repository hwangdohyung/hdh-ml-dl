# KERC22 Baseline

This code is distributed as a reference baseline for KERC2022 Emotion Recognition International Challenge.
This baseline is provided as an example for handling the dataset used in the competition.

Baseline Model performs the task of classification of the socio-behavioral emotional state 
(euphoria, dysphoria or neutral) of the speaker for each spoken sentence spoken in a conversation.

**NOTE:** This baseline code is only for reference. The validation part has been omitted from the code. 
The baseline was trained in the presence of validation set, which we use as public test in the first round of competition 
and the labels are not provided to the participants of KERC'22. 
Therefore, the baseline code only contains the training part...
Please split training set and use for validation.

## Requirements
The baseline is implemented in PyTorch and depends on several external packages mentioned in `requirements.txt`

## Dataset
KERC22 Dataset is a text-based conversation dataset,
distributed for use in the KERC2022 competition. For details, please refer to the Dataset description on provided page.


## Usage
### Preprocessing
With the dataset in `data_dir` configured in `conf/config.ini`, preprocessing involves two steps:
1. Prepare contextual information from the scene

` python prepare_context.py`

2. Extract BERT features for input to classification model

` python bert_ftrs.py`

### Classification

`python main.py`pyth

###  Generate Submission CSV file

`python generate_submission.py`

- Once the Private Test data is released, both public and private test results need to be submitted in a single CSV file.
  (Please check generate_submission.py file for generating submission file.)

