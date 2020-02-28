# Tuna
pytorch template for running deep learning experiments.
eventually, my goal is to provide as input a json file (with all relevant hyperparameters, choice of optimizer, neural network model, loss function, etc.) and it'll run the experiments for you automatically.

## Environment
`conda env create -f environment.yml` 

## Features to add:
- natural language processing experimentation (offload most of the work to allennlp)
- run.py file for ability to run multiple experiments at once (with a variety of hyperparameters basically)
- mean and variance information for datasets (utils)

