# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 00:49:47 2019

@author: ahmet
"""

import numpy as np
import binvox_rw

a = np.arange(0,400)

x_train = np.zeros(shape=(400,64,64,64), dtype=float)
liste = list()

for i in a:
    with open('train(' + str(i) + ').binvox','rb') as f:
        model = binvox_rw.read_as_3d_array(f)
        narray = np.ndarray(shape=(1,64,64,64), dtype=float)
        narray = model.data
        narray = narray.astype(float)
        x_train[i,:] = narray
        print(i)
        
    
np.save("x_train.npy",x_train)
        
