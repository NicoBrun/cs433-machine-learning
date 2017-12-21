"""Create a neural network as model"""
import numpy as np
import matplotlib
import os
import keras
from keras.models import Sequential,load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Activation, Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from sklearn import cross_validation
from road.realhelper.helpers import load_image, img_crop, label_to_img



def from_file(precomputed_model):
    return load_model(precomputed_model)

def create_model(X,Y,data_augmentation, patch_size):

    batch_size = 64
    epochs = 1
    split_ratio = 0.2
    channel_size = 128
    conv2D_size = 3
    maxPool2D_size = 2
    x_train, x_test, y_train, y_test = cross_validation.train_test_split(X, Y, test_size=split_ratio)

    # reformat in function the Keras input image dimensions
    img_rows, img_cols = patch_size, patch_size

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 3, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 3, img_rows, img_cols)
        input_shape = (3, img_rows, img_cols)
    else:
        input_shape = (img_rows, img_cols, 3)
    
    # construct the image generator for data augmentation
    aug = ImageDataGenerator(
            rotation_range=180,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='nearest')

    #layer 1
    model = Sequential()
    model.add(Conv2D(channel_size, (conv2D_size, conv2D_size), input_shape=input_shape))
    model.add(Activation('relu'))

    #layer 2
    model.add(Conv2D(channel_size, (conv2D_size, conv2D_size), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(maxPool2D_size, maxPool2D_size)))

    #layer 3
    model.add(Conv2D(channel_size, (conv2D_size, conv2D_size)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(maxPool2D_size, maxPool2D_size)))

    #
    model.add(Flatten())
    model.add(Dense(batch_size))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

    print("* description of our model:")
    model.summary()

    print("* start of our training")
    if not data_augmentation:
        H = model.fit(x_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                verbose=1,
                validation_data=(x_test, y_test))
    else:
        k = len(x_train) // batch_size
        H = model.fit_generator(aug.flow(x_train, y_train, batch_size=batch_size),
                        validation_data=(x_test, y_test), steps_per_epoch=k * 2,
                        epochs=epochs, verbose=1)

    return model

def predict_and_save_test_imgs(model, test_path, results_path, patch_size):
    imgs_patch = load_test_imgs(test_path, patch_size)
    predictions = model.predict(imgs_patch)
    saveTestImgsOutput(predictions, results_path, patch_size)

def load_test_imgs(test_path, patch_size):
    n=50
    test_directory = test_path+"/test_"
    imgs = [load_image(test_directory + str(i)+"/test_"+str(i)+".png") for i in range(1,n+1)]
    img_patches = [img_crop(imgs[i], patch_size, patch_size) for i in range(n)]
    img_patches = np.asarray([img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))])
    return np.asarray(img_patches)

def saveTestImgsOutput(predictions, results_path, patch_size):
    if not os.path.isdir(results_path):
        os.makedirs(results_path)
    w, h = 608, 608
    size = 1444*int((16/patch_size)**2)
    n = 50
    for i in range(n):
        predicted_im = label_to_img(w, h, patch_size, patch_size, predictions[size*i:size*(i+1)])
        name_to_save = results_path+'img_'+str(i+1)+'.png'
        H = np.zeros((w,h,3))
        H[predicted_im>0.5] = [1,1,1]
        H[predicted_im<0.5] = [0,0,0]
        matplotlib.image.imsave(name_to_save, H)