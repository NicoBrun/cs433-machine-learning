import numpy as np
from skimage.transform import rotate
import os,sys

class Data:

    def __init__(self, train_images_path, ground_truth_path, test_images_path):
        self.tr = train_images_path
        self.te = test_images_path
        self.gr = ground_truth_path

    def load_image(self, infilename):
        data = mpimg.imread(infilename)
        return data

    def correction(x):
        def corr(x, n):
            return (n - np.mod(x, n)) * np.mod(x, n) / n

        return corr(x, 45) + corr(x, 15) + corr(x, 9) + corr(x, 5) + corr(x, 3)

    def img_crop(im, w, h):
        list_patches = []
        imgwidth = im.shape[0]
        imgheight = im.shape[1]
        is_2d = len(im.shape) < 3
        for i in range(0, imgheight, h):
            for j in range(0, imgwidth, w):
                if is_2d:
                    im_patch = im[j:j + w, i:i + h]
                else:
                    im_patch = im[j:j + w, i:i + h, :]
                list_patches.append(im_patch)
        return list_patches

    def save(self, batch_size = 16, train = True, rotation_number = 1):
        if(train):
            files_im = os.listdir(self.tr)
            files_gr = os.listdir(self.gr)
            n= len(files_im)
            imgs = [self.load_image(self.tr + files_im[i]) for i in range(n)]
            gt_imgs = [self.load_image(self.gr + files_gr[i]) for i in range(n_gr)]
            #rotation
            rotation_angles = np.random.rand(n,rotation_number)*360;
            rotation_angles[:,0] = 0;
            imgs_rotate = np.empty([n,rotation_number])
            gts_rotate = np.empty([n,rotation_number])
            for j in np.arange(n):
                imgs_rotate[j,:] =[rotate(imgs[j],rotation_angles[j,i],resize = True) for i in range(rotation_number)]
                gts_rotate[j,:] = [rotate(gt_imgs[j],rotation_angles[j,i],resize = True) for i in range(rotation_number)]
                #zoom in the images to avoid black part due to rotation
                for j in np.arange(rotation_number):
                    size = imgs_rotate.shape[0]
                    index = int(np.floor(size*(1-rotation_angles[j,i]/90)*rotation_angles[j,i]/90)+correction(rotation_angles[j,i]))+int(rotation_angles[j,i] == 0)
                    imgs_rotate[j,i] = imgs_rotate[j,i][index:size-index,index:size-index]
                    gts_rotate[j,i] = gts_rotate[j,i][index:size-index,index:size-index]

            imgs_rotate = np.reshape(imgs_rotate,[1,n*rotation_number])
            gts_rotate = np.reshape(gts_rotate,[1,n*rotation_number])
            #patch separation
            img_patches = [img_crop(imgs_rotate[i], batch_size, batch_size) for i in range(n*rotation_number)]
            gt_patches = [img_crop(gts_rotate[i], batch_size, batch_size) for i in range(n*rotation_number)]
            # Linearize list of patches
            img_patches = np.asarray(
                [img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))])
            gt_patches = np.asarray(
                [gt_patches[i][j] for i in range(len(gt_patches)) for j in range(len(gt_patches[i]))])
            np.save("images_"+rotation_number+"_rotation_"+batch_size+"_patch.npz",img_patches)
            np.save("ground_truth_"+rotation_number+"_rotation_"+batch_size+"_patch.npz",gt_patches)
        else :
            files_test = os.listdir(self.te)
            n = len(files_test)
            tests = [self.load_image(self.te + files_test[i]) for i in range(n)]
            #rotation
            rotation_angles = np.random.rand(n, rotation_number) * 360;
            rotation_angles[:, 0] = 0;
            test_rotate = np.empty([n, rotation_number])
            for j in np.arange(n):
                test_rotate[j, :] = [rotate(tests[j], rotation_angles[j, i], resize=True) for i in
                                     range(rotation_number)]
                # zoom in the images to avoid black part due to rotation
                for j in np.arange(rotation_number):
                    size = test_rotate.shape[0]
                    index = int(
                        np.floor(size * (1 - rotation_angles[j, i] / 90) * rotation_angles[j, i] / 90) + correction(
                            rotation_angles[j, i])) + int(rotation_angles[j, i] == 0)
                    test_rotate[j, i] = test_rotate[j, i][index:size - index, index:size - index]
            test_rotate = np.reshape(test_rotate,[1,n*rotation_number])
            #patch separation
            test_patches = [img_crop(test_rotate[i], batch_size, batch_size) for i in range(n*rotation_number)]
            #Linerize list of patches
            test_patches =np.asarray(
                [test_patches[i][j] for i in range(len(test_patches)) for j in range(len(test_patches[i]))])
            np.save("test_"+rotation_number+"_rotation_"+batch_size+"_patch.npz",test_patches)


    def load(self,train = True,batch_size= 16, rotation_number=1):
        if(train):
            return np.load("images_"+rotation_number+"_rotation_"+batch_size+"_patch.npz"), np.load("ground_truth_"+rotation_number+"_rotation_"+batch_size+"_patch.npz")
        else :
            return np.load("test_"+rotation_number+"_rotation_"+batch_size+"_patch.npz")