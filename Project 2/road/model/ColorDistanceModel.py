import numpy as np
from PIL import Image
import skimage.filters
import skimage.morphology
import skimage.transform
import matplotlib.pyplot as plt

from Model import Model
class ColorDistanceModel(Model):
    def __init__(self, tolerance=5.0):
        self.tolerance = tolerance

    def predict(self, image):
        image = np.asarray(image, dtype=np.float32) / 255.0
        ref = np.array([124, 118, 113], dtype=np.float32) / 255.0

        distances = np.sqrt(((image - ref[np.newaxis, np.newaxis, :]) ** 2.0).sum(axis=2))

        probabilities = (1.0 - (distances - 0.1) * 5.0).clip(0.0, 1.0)

        #probabilities = np.arctan2(skimage.filters.sobel_h(probabilities), skimage.filters.sobel_v(probabilities))
        #probabilities = skimage.filters.gaussian(probabilities, sigma=5.0)

        shape = skimage.morphology.square(3)
        probabilities = skimage.morphology.opening(probabilities, shape)

        #probabilities = skimage.morphology.closing(probabilities, shape)





        return probabilities
