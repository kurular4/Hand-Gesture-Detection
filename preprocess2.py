# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 17:53:00 2020

@author: User
"""

import numpy as np
import os
import cv2
from random import shuffle

dir = r'datasethand'

training_data = []

for folder in os.listdir(dir):
    path = dir + '\\' + folder
    for img in os.listdir(path):
        label = str(folder)
        img_path = os.path.join(path,img)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if img is not None:
            img = cv2.resize(img, (100, 100))
            training_data.append([np.array(img), label])
        else:
            pass
        
shuffle(training_data)
np.save('data.npy', training_data)
