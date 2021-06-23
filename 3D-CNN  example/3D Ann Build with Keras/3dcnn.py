# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 01:21:51 2019

@author: ahmet
"""

#%%
#Load Numpy Arrays
import numpy as np
y = np.load("etiket_matrisim.npy")
x_train = np.load("x_train.npy")


#%%
import numpy as np
x_train_e = np.expand_dims(x_train,axis=4)

#%%

from sklearn.model_selection import train_test_split

x_train1, x_test, y_train, y_test = train_test_split(x_train_e,y,test_size = 0.2, random_state=1)

#%%

from keras.models import Sequential
from keras.layers import Input, Dense, Activation, Dropout, Flatten
from keras.layers import Conv3D, MaxPooling3D

#%%
input_shape = (64,64,64,1)

model = Sequential()
model.add(Conv3D(2,(6,6,6),input_shape=input_shape,padding="same"))
model.add(Activation("relu"))
model.add(MaxPooling3D(pool_size=(6,6,6),padding="same"))
model.add(Activation("relu"))

model.add(Flatten())

model.add(Dense(32))
model.add(Activation("relu"))
model.add(Dropout(0,3))

model.add(Dense(16))
model.add(Activation("relu"))
model.add(Dropout(0,3))

model.add(Dense(4))
model.add(Activation("softmax"))

model.compile(optimizer ="Adadelta", loss = "categorical_crossentropy", metrics=["accuracy"])
model.summary()

#%%

model.fit(x_train1,y_train, batch_size=32,epochs=10, validation_split = 0.2)

#%%

from keras.models import load_model
model.save('my_model.h5')

model = load_model('my_model.h5')