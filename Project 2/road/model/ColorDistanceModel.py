import numpy as np
from Model import Model
class ColorDistanceModel(Model):
    def __init__(self, tolerance=5.0):
        self.tolerance = tolerance

    def distance(self, rgb):
        mean_grey = 127
        r = (rgb[0] - 123)**2
        g = (rgb[1] - 119)**2
        b = (rgb[2] - 111)**2
        return np.sqrt(r + g + b) #* 20

    def dist_mean(self, col):
        if(col < 150):
            return 255
        else:
            return 0


    def predict(self, image):

        new_img = np.zeros(image.shape, dtype=np.float32)
        for i in range(len(image)):
            for j in range(len(image[0])):
                #print(image[i][j].shape)
                new_img[i][j] = self.distance(image[i][j])
        #return np.zeros(image.shape, dtype=np.float32)
        return new_img
