"""
module used to load images into memory
and return an array of patches of images, with its corresponding Y ={0,1}
"""
import os
import matplotlib.image as mpimg
import numpy as np
from road.realhelper.helpers import value_to_class, load_image, img_crop


def create_xy_from_patch(img_path, gt_path, patch_size):

    #load the images into memory as np array

    files = os.listdir(img_path)
    n = len(files)# Load maximum 100 images
    print("    Loading " + str(n) + " images")
    imgs = [load_image(img_path + files[i]) for i in range(n)]
    gt_imgs = [load_image(gt_path + files[i]) for i in range(n)]

    img_patches = [img_crop(imgs[i], patch_size, patch_size) for i in range(n)]
    gt_patches = [img_crop(gt_imgs[i], patch_size, patch_size) for i in range(n)]

    # Linearize list of patches
    img_patches = np.asarray([img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))])
    gt_patches =  np.asarray([gt_patches[i][j] for i in range(len(gt_patches)) for j in range(len(gt_patches[i]))])

    X = np.asarray(img_patches)
    Y = np.asarray([value_to_class(np.mean(gt_patches[i])) for i in range(len(gt_patches))])
    return X, Y

