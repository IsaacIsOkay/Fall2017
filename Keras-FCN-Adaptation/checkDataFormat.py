from keras.preprocessing.image import *
from keras.applications.imagenet_utils import preprocess_input
from keras import backend as K
from PIL import Image
import numpy as np
import os
import cv2
import sys
label_dir       = os.path.expanduser('~/.keras/datasets/weedspic/label_png/')

label_0 = Image.open(label_dir + '0000.png')
label_0 = img_to_array(label_0)
numOfZeros = 0
numOfOther = 0
for i in label_0:
   for k in i:
	if(k == 0):
	    numOfZeros +=1
        else:
            print(k)
print(numOfZeros)
print(numOfOther)
