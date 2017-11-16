# Machine Learning Project 1: finding the Higgs Boson

This project is about solving a binary classification problem. Given input data and a vector and the corresponding collision events, the goal is to determine a model that can predict the best output for new data.

## Getting Started

Git clone this repo and then run:
To get prediction file; in shell, go to project directory and then run
```
python run.py
```
The computation lasts for around 20 min on a i5 core
Its output is in 4jet_feat_process_30000.csv

To get the errors per model run
```
python all_models_1jet.py
python all_models_4jet.py
python all_models_no_processing_1jet.py
python all_models_no_processing_4jet.py
```

To visualize the data run
```
python visu_correlation.py
python cross_validation.py
```
The output of cross_validation is in cross_validation.png

All the files need the proj1_helpers.py, help_functions.py and implementations.py files to run.

### Prerequisites

The project needs some python libraries:
```
numpy
matplotlib
```

### Installing

All the files has to be in the same folder as train.csv and test.csv. Those files are not included in the repo.
