from road.helper.helpers import  load_image
import numpy as np
import matplotlib
from scipy import ndimage
import os,sys

def filter_thresh(image, threshold, kernel_w, kernel_h, squared=False, fill_holes=False):
    kernel = np.ones((kernel_h, kernel_w))
    if (squared and kernel_w == kernel_h and np.mod(kernel_w,2)==1):
        mid = int(np.floor(kernel_w / 2))
        kernel[mid, mid] = 5
        if (fill_holes):
            kernel[mid, mid] = 0

    image = ndimage.filters.convolve(image, kernel)
    image[image < threshold] = 0
    image[image >= threshold] = 1

    return image

def resize(imgs,new_size,n,pixel_size):
    imgs_resized = np.zeros((n, new_size, new_size))
    s = len(imgs[0])
    print(s)
    for r in range(n):
        for j in range(s):
            for k in range(s):
                imgs_resized[r][j * pixel_size:(j + 1) * pixel_size, k * pixel_size:(k + 1) * pixel_size] = imgs[r][
                    j, k]
    return imgs_resized

def post_process(prediction_file_name, save_file_name):
    if not os.path.isdir(save_file_name):
        os.makedirs(save_file_name)

    images_dir = prediction_file_name
    files = os.listdir(images_dir)
    n = len(files)

    pixel_size = 16
    imgs_big = np.array([load_image(images_dir + files[i]) for i in range(n)])
    original_size = len(imgs_big[0])
    #keep only one pixel every pixel_size to have real pixel of size 1x1
    imgs = [imgs_big[i][::pixel_size, ::pixel_size] for i in range(n)]

    # reduce the dimension from (3,38,38) to (38,38)
    imgs_pure = np.mean(imgs, -1)
    imgs_pure[imgs_pure == 0.25] = 0

    # remove noise
    imgs = [filter_thresh(imgs_pure[i], 7, 5, 5, squared=True) for i in range(n)]
    #keep only the one that were road before
    imgs = np.array([imgs[i] + imgs_pure[i] for i in range(n)])
    imgs[imgs < 2] = 0
    imgs[imgs == 2] = 1

    imgs_noiseless = imgs

    # fill holes
    imgs = [filter_thresh(imgs[i], 7, 3, 3, squared=True, fill_holes=True) for i in range(n)]
    imgs = np.array([imgs[i] + imgs_noiseless[i] for i in range(n)])
    imgs[imgs < 1] = 0
    imgs[imgs >= 1] = 1

    imgs_noise_hole = imgs

    imgs = [filter_thresh(imgs[i], 7, 3, 3, squared=True, fill_holes=True) for i in range(n)]
    imgs = np.array([imgs[i] + imgs_noise_hole[i] for i in range(n)])
    imgs[imgs < 1] = 0
    imgs[imgs >= 1] = 1

    #resize the images to their original size
    imgs = resize(imgs,original_size,n,pixel_size)

    #save img
    for i in range(n):
        name = save_file_name + files[i]
        matplotlib.image.imsave(name, imgs[i], cmap="Greys_r")

