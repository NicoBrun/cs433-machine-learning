#%matplotlib inline
from ColorDistanceModel import ColorDistanceModel
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
"""files = os.listdir(image_dir)
img = load_image(image_dir + files[7])

model = ColorDistanceModel()

new_img = model.predict(img)

# Show first image and its model image
cimg = concatenate_images(img, new_img)
fig1 = plt.figure(figsize=(10, 10))

plt.imshow(cimg, cmap='Greys_r')
plt.show()"""
index = 8
path = "../../training/images/satImage_%03d.png" % index
truth_path = "../../training/groundtruth/satImage_%03d.png" % index

image = Image.open(path)
image = np.asarray(image)

truth = Image.open(truth_path).convert('RGB')
truth = np.asarray(truth)[:, :, 0] / 255.0
tmp = np.zeros([truth.shape[0], truth.shape[1], 4], dtype=np.float32)
tmp[:, :, 0] = truth
tmp[:, :, 3] = 0.2
truth = tmp

model = ColorDistanceModel()

probabilities = model.predict(image)

# Show first image and its model image
cimg = concatenate_images(image, probabilities)
fig1 = plt.figure(figsize=(10, 10))

plt.imshow(cimg, cmap='Greys_r')
plt.show()
# probabilities = skimage.filters.sobel(probabilities)



"""fig, axes = plt.subplots(1, 2)
axes[0].set_aspect('equal')
axes[1].set_aspect('equal')
axes[0].imshow(image, interpolation='nearest')
axes[1].imshow(probabilities, interpolation='nearest', cmap='gray')
axes[1].imshow(truth, interpolation='nearest')
# h, theta, d = skimage.transform.hough_line(probabilities)
# for _, angle, dist in zip(*skimage.transform.hough_line_peaks(h, theta, d)):
  # y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
  # y1 = (dist - image.shape[1] * np.cos(angle)) / np.sin(angle)
  # axes[1].plot((0, image.shape[1]), (y0, y1), '-r')
axes[1].set_xlim([0,image.shape[0]])
axes[1].set_ylim([0,image.shape[1]])
plt.show()"""
