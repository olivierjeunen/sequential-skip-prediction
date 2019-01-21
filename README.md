# WSDM Cup 2019: Spotify Sequential Skip Prediction Challenge - 5th Place Solution
This repository contains the source code for our approach to the 2019 WSDM Cup Spotify Sequential Skip Prediction Challenge.
We achieved the 5th position under the 'Adrem Data Lab' team. More information on the competition, as well as the dataset, can be found [here](https://www.crowdai.org/challenges/spotify-sequential-skip-prediction-challenge).

Our solution comprises a modular, scalable and data-efficient pipeline, a minimum of feature engineering and a single recurrent neural network architecture trained with a custom weighted loss function.
A detailed motivation and description of our approach can be found in the [accompanying paper](http://adrem.uantwerpen.be//bibrem/pubs/WSDMCupJeunen2019.pdf).

The current script expects a GPU installation of Tensorflow, with Keras on top.
Furthermore, model training is parallellised over 2 GPUs.

## Setup
The main script expects a data-folder containing the gzipped Spotify dataset in `training_set`, `test_set` and `track_features` subfolders, along with empty `preprocessed`, `predictions` and `submissions` subfolders. These latter 3 will hold intermediate (gzipped preprocessed matrices for the training and test set) and output (raw predictions and CrowdAI submission format) files.

## Python Packages
The required python packages can be found in `requirements.txt`.
Using a package manager such as `pip`, they can easily be installed with the following command:

`pip3 install -r requirements.txt`

## Usage
`python3 RNN.py <path> <start_day> <end_day> <sub_ID>`
- `<path>`: the path to a data-folder containing the Spotify dataset. Intermediate and output files will be written here.
- `<start_day>`: the day to start processing on (inclusive, integer between 0 - 65).
- `<end_day>`: the day to end processing with (exclusive, integer between 1 - 66).
- `<sub_ID>`: Unique identifier for this submission, used to avoid overwriting files between different runs.

<img src="https://research.yahoo.com/mobstor/logo_wsdm_med.png" width="33%" height="33%"><img src="https://storage.googleapis.com/pr-newsroom-wp/1/2018/11/Spotify_Logo_CMYK_Green.png" width="37%" height="37%">
