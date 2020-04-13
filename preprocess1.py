# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 12:09:59 2020

@author: User
"""

import os
import cv2
from detecthand import detect_hand_by_dir

for i in range(11):
    dirname = r'datasethand\G{0}'.format(i+1)
    os.mkdir(dirname)
    for j in range(30):
        cropped_img = detect_hand_by_dir(r'G{0}\{1}-color.png'.format(i+1, j+1))
        cv2.imwrite(r'{0}\img{1}.png'.format(dirname, j+1), cropped_img)
    

