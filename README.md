# Evaluation of real-time tumor contour prediction using LSTM networks for MR-guided radiotherapy
Framework to train, validate and test three different LSTM networks for 2D tumor contour prediction 
and compare their performance with a  baseline no-predictor.

Elia Lombardo\
LMU Munich\
Elia.Lombardo@med.uni-muenchen.de

Evaluated with Python version 3.8.13 and PyTorch version 1.12.0

## Installation
* Download the repository to a local folder of your preference or clone the repository.
* Either (a) create an anaconda virtual environment and run `pip3 install -r requirements.txt` or (b) build a Docker 
image based on the provided `Dockerfile` and run a container while mounting the `lstm_for_centroid_prediction` folder (recommended).
* Open `lstm_contour_prediction/code/config.py` and change `path_project` to your local path to the `lstm_contour_prediction` folder.

## Usage
* In the file  `lstm_contour_prediction/code/config.py` you can set all the options for your
models, e.g. whether to perform training or testing, whether to run offline or online models, and all the
hyper-parameter settings. 
* After that, run the corresponding main script in the terminal. 
For instance `python main_train_val_model.py` to run an offline optimization of the LSTM or `python main_online_no_predictor.py` 
to evaluate the no-predictor.
* The weights of the best models, metrics histories, etc. will be saved under `lstm_contour_prediction/results`.

## Publication
If you use this code in a scientific publication, please cite our paper:
TBA
