#%matplotlib inline
from ColorDistanceModel import ColorDistanceModel
import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import os,sys
from PIL import Image


def load_image(infilename):
    data = mpimg.imread(infilename)
    return data

def concatenate_images(img, gt_img):
    nChannels = len(gt_img.shape)
    w = gt_img.shape[0]
    h = gt_img.shape[1]
    if nChannels == 3:
        cimg = np.concatenate((img, gt_img), axis=1)
    else:
        gt_img_3c = np.zeros((w, h, 3), dtype=np.uint8)
        gt_img8 = img_float_to_uint8(gt_img)
        gt_img_3c[:,:,0] = gt_img8
        gt_img_3c[:,:,1] = gt_img8
        gt_img_3c[:,:,2] = gt_img8
        img8 = img_float_to_uint8(img)
        cimg = np.concatenate((img8, gt_img_3c), axis=1)
    return cimg

image_dir = "../../training/images/"
files = os.listdir(image_dir)
img = load_image(image_dir + files[7])

model = ColorDistanceModel()

new_img = model.predict(img)

# Show first image and its model image
cimg = concatenate_images(img, new_img)
fig1 = plt.figure(figsize=(10, 10))

plt.imshow(cimg, cmap='Greys_r')
plt.show()
