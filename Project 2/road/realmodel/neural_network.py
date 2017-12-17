"""Create a neural network as model"""
import keras
from keras.models import load_model
from road.realhelper.helpers import load_image, img_crop, label_to_img
import numpy as np
import matplotlib
import os

def from_file(precomputed_model):
    return load_model(precomputed_model)

def create_model(X,Y,data_augmentation):
    pass

def predict_one_img(img_path,id):
    pass

def predict_and_save_test_imgs(model, test_path, results_path, patch_size):
    imgs_patch = load_test_imgs(test_path, patch_size)
    predictions = model.predict(imgs_patch)
    saveTestImgsOutput(predictions, results_path, patch_size)

def load_test_imgs(test_path, patch_size):
    #1444 tiles per image
    n=50
    test_directory = test_path+"/test_"
    imgs = [load_image(test_directory + str(i)+"/test_"+str(i)+".png") for i in range(1,n+1)]
    img_patches = [img_crop(imgs[i], patch_size, patch_size) for i in range(n)]
    img_patches = np.asarray([img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))])
    return np.asarray(img_patches)

def saveTestImgsOutput(predictions, results_path, patch_size):
    if not os.path.isdir(results_path):
        os.makedirs(results_path)
    w = 608
    h = 608
    size = 1444
    n = 50
    for i in range(n):
        predicted_im = label_to_img(w, h, patch_size, patch_size, predictions[size*i:size*(i+1)])
        name_to_save = results_path+'img_'+str(i+1)+'.png'
        H = np.zeros((w,h,3))
        H[predicted_im>0.5] = [1,1,1]
        H[predicted_im<0.5] = [0,0,0]
        matplotlib.image.imsave(name_to_save, H)