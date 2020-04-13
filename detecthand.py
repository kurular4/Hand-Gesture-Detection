# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 18:51:49 2020

@author: User
"""
import cv2
import numpy as np

def detect_hand_by_dir(img_path):
    img = cv2.imread(img_path)
    return detect_hand(img)

# simple hand detection assuming most occupying object in the image is hand    
def detect_hand(img):
    img_ycbcr = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

    skin_min = np.array([0, 135, 85], np.uint8)
    skin_max = np.array([255, 180, 135], np.uint8)
    img_thrs = cv2.inRange(img_ycbcr, skin_min, skin_max)

    contours, hierarchy = cv2.findContours(img_thrs, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours = max(contours, key = cv2.contourArea)

    x,y,w,h = cv2.boundingRect(contours)
    
    cropped_img = img[y:y+h, x:x+w]

    return x, y, w, h, cropped_img
