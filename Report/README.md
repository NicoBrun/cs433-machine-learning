# MachineLearningLesBGS

## TODO:
* remplir le readme
* ranger le dossier du projet, (un dossier visualisation pour les histogrammes, etc)
* refaire un rendu kaggle
* pr�parer les fichier que l'on envoye dans un zip (on envoye pas les csv ou le milliard d'image?)
* finir le rapport


## Done
* changer les 6 functions models (celle voulue par le project description doivent retourner (w,loss))
* run les 6 models et avoir les erreures et les ajouter dans le rapport comme justification
* run avec et sans le 4 jets, ajouter les r�sultat au rapport
* run avec et sans les features processing, ajouter les r�sults au rapports
* faire une image de correlation pour expliquer la feature distance
* finir cross-validation (10^-5, 10^-5)
* run cross-validation pour trouver lambda/gamma cool
* run 4 jets avec le bon model
* faire graphique des lambda/gamma cool et les ajouter au rapport
* mettre le code que l'on veut dans le run.py
* commenter tout le code
* enlever les print dans implementation.py

## Description of the project

## Main file
Run run.py to get the submission file. It has to be in the same folder as train.csv and test.csv. The output is 4jet_feat_process_30000.csv

## Error files

## Helper files

## Visualization files

# Machine Learning Project 1: finding the Higgs Boson

This project is about solving a binary classification problem. Given input data and a vector and the corresponding collision events, the goal is to determine a model that can predict the best output for new data.

## Getting Started

Git clone this repo and then run:
To get prediction file; in shell, go to project directory and then run
```
python run.py
```
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
