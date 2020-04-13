# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 15:44:52 2020

@author: User
"""

import numpy as np
import tensorflow as tf
import cv2
from tensorflow.python.keras.models import load_model
from detecthand import detect_hand

model = load_model('model.h5')

video_capture = cv2.VideoCapture(0)

labels = {
        0: 'palm',
        1: 'num_two',
        2: 'num_three',
        3: 'palm_close_finger',
        4: 'fist',
        5: 'rock',
        6: 'index_finger',
        7: 'palm_open_finger',
        8: 'ok',
        9: 'thumbs_up',
        10: 'little_finger' }

while True:
    ret, frame = video_capture.read()
    x, y, w, h, framedetected = detect_hand(frame)
    
    frameformatted = cv2.cvtColor(framedetected, cv2.COLOR_BGR2GRAY)
    frameformatted = cv2.resize(frameformatted, (100,100))
    frameformatted = tf.cast(frameformatted[np.newaxis,:,:,np.newaxis], tf.float32)
    
    prediction = model.predict(frameformatted)
    
    cv2.rectangle(frame, (x, y), (x + w, y + h), (25,255,50), 1)
    cv2.putText(frame, labels[np.argmax(prediction)], (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255))
    
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()


