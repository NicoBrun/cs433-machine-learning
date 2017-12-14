#%matplotlib inline
from ColorDistanceModel import ColorDistanceModel
from PostProcessModel import PostProcessModel
import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import os,sys
from PIL import Image

def img_float_to_uint8(img):
    PIXEL_DEPTH = 255
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * PIXEL_DEPTH).round().astype(np.uint8)
    return rimg

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


model = ColorDistanceModel()

train_path = "../../training/"

images = []
truths = []

for p in os.listdir(train_path + "images"):
    image = Image.open(train_path + "images/" + p)
    truth = Image.open(train_path + "groundtruth/" + p)
    images.append(np.asarray(image))
    t = np.asarray(truth)#.clip(0.0, 1.0)
    G = t
    G.setflags(write = True)
    G[t < 127] = 0
    G[t >= 127] = 1
    truths.append(G)

mean = model.fit(images, truths)

processed_images = []
for i in images:
    prob = model.predict(i, mean)
    prob = prob.reshape(prob.shape[0], prob.shape[1], 1)
    new_img = np.append(i, prob, axis = 2)
    processed_images.append(np.asarray(new_img))

print(np.asarray(processed_images).shape)
image = images[0]
probabilities = model.predict(image, mean)

cimg = concatenate_images(image, probabilities)
fig1 = plt.figure(figsize=(10, 10))

plt.imshow(cimg, cmap='Greys_r')
plt.show()
# probabilities = skimage.filters.sobel(probabilities)
