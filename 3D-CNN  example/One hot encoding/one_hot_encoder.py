# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 23:34:46 2019

@author: ahmet
"""

import numpy as np
from keras.utils import np_utils

number_classes = 4

number_samples = 400

labels = np.ones((number_samples,),dtype="int")

labels[0:100]=0
labels[100:200]=1
labels[200:300]=2
labels[300:400]=3

# names = 0bathtub, 1bed, 2chair, 3monitor

Y = np_utils.to_categorical(labels, number_classes)

y = np.save("etiket_matrisim.npy",Y)