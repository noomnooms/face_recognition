import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.imagenet_utils import preprocess_input

def resize(image, size:tuple):
    '''
    resize image to model input size 

    ### parameters

    image: arraytype

    size: (dim, dim)
    '''
    img = cv2.resize(image, size)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

def findEuclideanDistance(x, y):
    return np.sqrt(np.sum((x - y) ** 2))

def findCosineSimil(x, y):
    return 1 - np.dot(x, y) / (np.sqrt(np.dot(x, x)) * np.sqrt(np.dot(y, y)))

def L2_normalize(x):
    return x / np.sqrt(np.sum(np.multiply(x, x)))