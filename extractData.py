
import keras    
import glob
from tqdm import tqdm
import os
import cv2
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from keras.models import Model
from keras.layers import Input
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from matplotlib import pyplot as plt
import arabic_reshaper
from bidi.algorithm import get_display

def ID_convertor(ID_output):
    ID = ''
    for label in ID_output:
        ID += str(label)
    return ID
    
# letter_model = keras.models.load_model('letter_model')
digit_model = keras.models.load_model('digit_model')

#ID
ID_output = []
testPaths = glob.glob("ID*.jpg")
testPaths.sort()
for i, testPath in enumerate(testPaths):
    image = load_img(testPath, target_size=(28, 28), grayscale=True)
    image = img_to_array(image) / 255.
    orig_img = image.copy()
    image = np.expand_dims(image, 0)
    predictions = digit_model.predict(image)[0]
    label = np.argmax(predictions)
    proba = np.max(predictions)
    output = cv2.resize(orig_img, (400, 400))
    ID_output.append(label)
    # print("{}: {:.2f}%".format(label, proba * 100))
    # cv2.imshow(str(label), output)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

# #FN
# FN_output = []
# testPaths = glob.glob("FN*.jpg")
# testPaths.sort()
# for i, testPath in enumerate(testPaths):
#     image = load_img(testPath, target_size=(28, 28), grayscale=True)
#     image = img_to_array(image) / 255.
#     orig_img = image.copy()
#     image = np.expand_dims(image, 0)
#     predictions = letter_model.predict(image)[0]
#     label = np.argmax(predictions)
#     proba = np.max(predictions)
#     output = cv2.resize(orig_img, (400, 400))
#     FN_output.append(label)
#     # print("{}: {:.2f}%".format(label, proba * 100))
#     # cv2.imshow(str(label), output)
#     # cv2.waitKey()
#     # cv2.destroyAllWindows()
  
# #LN
# LN_output = []
# testPaths = glob.glob("LN*.jpg")
# testPaths.sort()
# for i, testPath in enumerate(testPaths):
#     image = load_img(testPath, target_size=(28, 28), grayscale=True)
#     image = img_to_array(image) / 255.
#     orig_img = image.copy()
#     image = np.expand_dims(image, 0)
#     predictions = letter_model.predict(image)[0]
#     label = np.argmax(predictions)
#     proba = np.max(predictions)
#     output = cv2.resize(orig_img, (400, 400))
#     LN_output.append(label)
#     # print("{}: {:.2f}%".format(label, proba * 100))
#     # cv2.imshow(str(label), output)
#     # cv2.waitKey()
#     # cv2.destroyAllWindows()  

print(ID_convertor(ID_output))
