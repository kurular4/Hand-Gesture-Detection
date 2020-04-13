# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 15:16:31 2020

@author: User
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder  
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten

data = np.load('data.npy', allow_pickle=True)

x = np.array([i[0] for i in data])
x = x.reshape(x.shape[0], 100, 100, 1)

y = [i[1] for i in data]
y = np.array(y)

train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3,
                                                    random_state=42)

le = LabelEncoder()
train_y = le.fit_transform(train_y)
test_y = le.fit_transform(test_y)


model = Sequential()

model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(100,100,1)))
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(11, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(train_x, train_y, validation_data=(test_x, test_y), epochs=12)

model.save('model.h5') 
np.save('test', test_x)






