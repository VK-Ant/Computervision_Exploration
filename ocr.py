#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 09:56:50 2023

@author: vk
"""



# install easyocr pytorch base library

#pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
#pip install easyocr

import easyocr as es
import cv2
from matplotlib import pyplot as plt
import numpy as np

IMAGE_PATH = "/home/vk/Downloads/drive-download-20231009T040720Z-001/IC15_test/word_1217.png"

reader = es.Reader(['en'],gpu=False)
result = reader.readtext(IMAGE_PATH)
print(result)

#initial input image show
img = cv2.imread(IMAGE_PATH)
plt.imshow(img)
plt.show()

#draw results

top_left = tuple(result[0][0][0])
bottom_right = tuple(result[0][0][2])
text = result[0][1]
font = cv2.FONT_HERSHEY_SIMPLEX

#single line

def single_line():
    img = cv2.imread(IMAGE_PATH)
    img = cv2.rectangle(img,top_left,bottom_right,(0,255,0),3)
    img = cv2.putText(img,text,top_left, font, 1,(255,0,0),3,cv2.LINE_AA)
    cv2.imwrite('output_ocr.png',img)
    plt.imshow(img)
    plt.show()

def multiple_line(IMAGE_PATH=IMAGE_PATH):
    #multiple line
    img = cv2.imread(IMAGE_PATH)
    spacer = 200
    for detection in result: 
        top_left = tuple(detection[0][0])
        bottom_right = tuple(detection[0][2])
        text = detection[1]
        img = cv2.rectangle(img,top_left,bottom_right,(0,255,0),3)
        img = cv2.putText(img,text,(10,spacer), font,1,(255,0,0),3,cv2.LINE_AA)
        spacer+=25
        
    plt.imshow(img)
    plt.show()
    
#single_line()
#multiple_line()