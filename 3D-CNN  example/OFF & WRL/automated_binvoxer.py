# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 00:15:43 2019

@author: ahmet
"""

import os
os.chdir(r'S:\udemy\3dcnn\cnn_egitim\train')
print(os.getcwd())
import time

k = 0
while k < 400:
    print(k)
    
    command = r"binvox -c -d 64 -t binvox -cb train(" + str(k) + ").off"
    os.popen(command)
    time.sleep(2)
    k = k + 1
    
    
    