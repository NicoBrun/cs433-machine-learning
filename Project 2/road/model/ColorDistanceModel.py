import numpy as np
from PIL import Image
import skimage.filters
import skimage.morphology
import skimage.transform
import matplotlib.pyplot as plt

from Model import Model
class ColorDistanceModel(Model):
    #mean = [0, 0, 0]
    def __init__(self, tolerance=5.0):
        self.tolerance = tolerance

    def fit(self, images, truths):
        sum = [0, 0, 0]
        total = [0, 0, 0];
        for i in range(len(images)):
            image = images[i]
            truth = truths[i]

            tbR = (image[:, :, 0] * truth).flatten()
            newR = tbR[np.nonzero(tbR)]
            tbG = (image[:, :, 1] * truth).flatten()
            newG = tbG[np.nonzero(tbG)]
            tbB = (image[:, :, 2] * truth).flatten()
            newB = tbB[np.nonzero(tbB)]

            sum = np.add(sum, [newR.sum(), newG.sum(), newB.sum()])
            total = np.add(total, [len(newR), len(newG), len(newB)])

        mean = np.divide(sum , total)

        return mean

    def predict(self, image, mean):
        image = np.asarray(image, dtype=np.float32) / 255.0
        ref = mean / 255.0

        distances = np.sqrt(((image - ref[np.newaxis, np.newaxis, :]) ** 2.0).sum(axis=2))

        probabilities = (1.0 - (distances - 0.1) * 5.0).clip(0.0, 1.0)
        shape = skimage.morphology.square(3)
        probabilities = skimage.morphology.opening(probabilities, shape)
        return probabilities
