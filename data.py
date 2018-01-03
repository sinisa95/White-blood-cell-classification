import cv2
import numpy as np
from os import listdir
from scipy.misc import imresize
from computer_vision import findWBC

def loadImages(folder, save=False):
    X = []
    Y = []
    for wbcType in listdir(folder):
        for imgName in listdir(folder +'/'+ wbcType):
            image = cv2.imread(folder +'/'+ wbcType + '/' + imgName)
            image = imresize(arr=image, size=(240, 320, 3))
            if image is not None:
                if save:
                    image = removeBlackPixels(image);
                    image = findWBC(image)

                    cv2.imwrite("processed_" + folder + '/' + wbcType + '/' + imgName, image)
                imageArray = np.asarray(image)
                X.append(imageArray)
                Y.append(wbcType)

    X = np.asarray(X)
    Y = np.asarray(Y)
    return X,Y

def removeBlackPixels(image):
    x,y,z = image.shape
    for i in range(0,x):
        for j in range(0,y):
            if image[i,j,0] < 30 and image[i,j,1]< 30 and image[i,j,2]<30:
                image[i,j,:]=255
    return image



