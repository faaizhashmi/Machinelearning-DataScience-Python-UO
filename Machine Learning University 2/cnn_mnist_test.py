#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 14:51:48 2019

@author: shaggy7000
"""

from keras.models import model_from_json
import numpy as np
from skimage import io


img=io.imread('five.png')
img=img[:,:,0]
io.imshow(img)

img=img.reshape(1, 28, 28, 1)
#img = np.expand_dims(img, axis = 0)
#img = np.expand_dims(img, axis = -1)


# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
 
ynew = loaded_model.predict_classes(img)
print("preditd number",ynew)

