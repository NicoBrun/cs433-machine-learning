# Machine Learning Project 2: road Segmentation

This project is about solving a binary classification problem. Given an image, we have to determine if every patch of 16x16 pixels in this image is either a road or not {1 or 0}.

### Prerequisites

The project needs some python libraries:
```
numpy
matplotlib
os
sys
Image
math
```

This project needs also a deep learning library named [Keras](keras.io) and this library run on top of [Tensorflow](https://www.tensorflow.org/).

To install those 2 externals libraries, run in a shell:
```
pip install keras
pip install tensorflow
```


### Installing

Exctract the zip file, you should have the file "run.py" and the folders named "road" and "saved_model". 

We provide an already trained model inside the "saved_model" if you don't want to to recrate a model from scratch.

You should have also the 2 zip files from kaggle.com containing the training and the test set.
Extract them into the same folder; the directory contains now 4 folders named "road", "saved_model", "training" and "test_set_images".

You are ready!


## Getting Started

To get the prediction file; in shell, go to project directory and then run

```
python run.py
```

It will load the precomputed model and use it to create the images prediction in the folder "test_set_result_model_final".
Then it will create the prediction file from this set of image and save it under "model_final.csv"

## Create the model from scratch

Instead of loading the model from "saved_model", we can create a new one from the training picture set.
In this case open run.py and change the value of load_model to False.
You can also choose if you want data augmentation and post processing by setting the boolean parameter.

Creating the model from scrath took us around +-20h, with data augmentation.