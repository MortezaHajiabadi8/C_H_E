
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

def FN_letter_convertor(output):
    string = ''
    output = output[::-1]
    output = output[:5]
    print(output)
    for item in output:
        if str(item) == '0':
            string += 'ا'
        elif str(item) == '1':
            string += 'ب'
        elif str(item) == '2':
            string += 'پ'
        elif str(item) == '3':
            string += 'ت'
        elif str(item) == '4':
            string += 'ث'
        elif str(item) == '5':
            string += 'ج'
        elif str(item) == '6':
            string += 'چ'
        elif str(item) == '7':
            string += 'ح'
        elif str(item) == '8':
            string += 'خ'
        elif str(item) == '9':
            string += 'د'
        elif str(item) == '10':
            string += 'ذ'
        elif str(item) == '11':
            string += 'ر'
        elif str(item) == '12':
            string += 'ز'
        elif str(item) == '13':
            string += 'ژ'
        elif str(item) == '14':
            string += 'س'
        elif str(item) == '15':
            string += 'ش'
        elif str(item) == '16':
            string += 'ص'
        elif str(item) == '17':
            string += 'ض'
        elif str(item) == '18':
            string += 'ط'
        elif str(item) == '19':
            string += 'ظ'
        elif str(item) == '20':
            string += 'ع'
        elif str(item) == '21':
            string += 'غ'
        elif str(item) == '22':
            string += 'ف'
        elif str(item) == '23':
            string += 'ق'
        elif str(item) == '24':
            string += 'ک'
        elif str(item) == '25':
            string += 'گ'
        elif str(item) == '26':
            string += 'ل'
        elif str(item) == '27':
            string += 'م'
        elif str(item) == '28':
            string += 'ن'
        elif str(item) == '29':
            string += 'و'
        elif str(item) == '30':
            string += 'ه'
        elif str(item) == '31':
            string += 'ی'
    print(get_display(arabic_reshaper.reshape(string)))

def LN_letter_convertor(output):
    string = ''
    output = output[::-1]
    output = output[:]
    print(output)
    for item in output:
        if str(item) == '0':
            string += 'ا'
        elif str(item) == '1':
            string += 'ب'
        elif str(item) == '2':
            string += 'پ'
        elif str(item) == '3':
            string += 'ت'
        elif str(item) == '4':
            string += 'ث'
        elif str(item) == '5':
            string += 'ج'
        elif str(item) == '6':
            string += 'چ'
        elif str(item) == '7':
            string += 'ح'
        elif str(item) == '8':
            string += 'خ'
        elif str(item) == '9':
            string += 'د'
        elif str(item) == '10':
            string += 'ذ'
        elif str(item) == '11':
            string += 'ر'
        elif str(item) == '12':
            string += 'ز'
        elif str(item) == '13':
            string += 'ژ'
        elif str(item) == '14':
            string += 'س'
        elif str(item) == '15':
            string += 'ش'
        elif str(item) == '16':
            string += 'ص'
        elif str(item) == '17':
            string += 'ض'
        elif str(item) == '18':
            string += 'ط'
        elif str(item) == '19':
            string += 'ظ'
        elif str(item) == '20':
            string += 'ع'
        elif str(item) == '21':
            string += 'غ'
        elif str(item) == '22':
            string += 'ف'
        elif str(item) == '23':
            string += 'ق'
        elif str(item) == '24':
            string += 'ک'
        elif str(item) == '25':
            string += 'گ'
        elif str(item) == '26':
            string += 'ل'
        elif str(item) == '27':
            string += 'م'
        elif str(item) == '28':
            string += 'ن'
        elif str(item) == '29':
            string += 'و'
        elif str(item) == '30':
            string += 'ه'
        elif str(item) == '31':
            string += 'ی'
    print(get_display(arabic_reshaper.reshape(string)))

def cal_acc(ID_output, FN_output, LN_output):
    ID_output = np.array(ID_output[:])
    FN_output = np.array(FN_output[3:])
    LN_output = np.array(LN_output[:])
    print(ID_output)
    ID = np.array([0,9,6,2,6,1,4,3])
    print(ID)
    print("Accuracy ID: ", (np.sum(ID_output == ID) / len(ID_output)) * 100, "%")
    FN = np.array([31,17,3,11,27])
    print(FN_output)
    print(FN)
    print("Accuracy FN: ", (np.sum(FN_output == FN) / len(FN_output)) * 100, "%")
    LN = np.array([9,0,1,0,31,5,0,7])
    print(LN_output)
    print(LN)
    print("Accuracy LN: ", (np.sum(LN_output == LN) / len(LN_output)) * 100, "%")
    
    

letter_model = keras.models.load_model('seq_letter_model')
digit_model = keras.models.load_model('seq_digit_model')


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

#FN
FN_output = []
testPaths = glob.glob("FN*.jpg")
testPaths.sort()
for i, testPath in enumerate(testPaths):
    image = load_img(testPath, target_size=(28, 28), grayscale=True)
    image = img_to_array(image) / 255.
    orig_img = image.copy()
    image = np.expand_dims(image, 0)
    predictions = letter_model.predict(image)[0]
    label = np.argmax(predictions)
    proba = np.max(predictions)
    output = cv2.resize(orig_img, (400, 400))
    FN_output.append(label)
    # print("{}: {:.2f}%".format(label, proba * 100))
    # cv2.imshow(str(label), output)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
  
#LN
LN_output = []
testPaths = glob.glob("LN*.jpg")
testPaths.sort()
for i, testPath in enumerate(testPaths):
    image = load_img(testPath, target_size=(28, 28), grayscale=True)
    image = img_to_array(image) / 255.
    orig_img = image.copy()
    image = np.expand_dims(image, 0)
    predictions = letter_model.predict(image)[0]
    label = np.argmax(predictions)
    proba = np.max(predictions)
    output = cv2.resize(orig_img, (400, 400))
    LN_output.append(label)
    # print("{}: {:.2f}%".format(label, proba * 100))
    # cv2.imshow(str(label), output)
    # cv2.waitKey()
    # cv2.destroyAllWindows()  

print(ID_convertor(ID_output))
FN_letter_convertor(FN_output)
LN_letter_convertor(LN_output)
print("************************************************")
cal_acc(ID_output, FN_output, LN_output)