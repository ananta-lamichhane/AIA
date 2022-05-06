#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Simon Matern
"""

import numpy as np
import cv2
import utils


def binarizeImage(img, thresh):
    """
    Given a coloured image and a threshold binarizes the image.
    Values below thresh are set to 0. All other values are set to 255
    """
    #Convert to grayscale and then use cv2.threshold().

    im_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    th, img= cv2.threshold(im_gray, thresh, 255, cv2.THRESH_BINARY)

    



    return img

def smoothImage(img):    
    """
    Given a coloured image apply a blur on the image, e.g. Gaussian blur
    """
    img = cv2.GaussianBlur(img,(11,11),0)
    return img

def doSomething(img):
    """
    Given a coloured image apply any image manipulation. Be creative!
    """
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


    kernel_sharpening = np.array([[-1,-1,-1], 
                              [-1,9,-1], 
                              [-1,-1,-1]])
    img = cv2.filter2D(image, -1, kernel_sharpening)
    
    cv2.circle(img, (img.shape[1]//2,img.shape[0]//2 ),50, (255,0,0), thickness=-1)
    
    return img


def processImage(img):
    """
    Given an coloured image applies the implemented smoothing and binarization.
    """
    # TODO
    img = smoothImage(img)
    img = binarizeImage(img, 125)
    return img


if __name__=="__main__":
    img = cv2.imread("test.jpg")
    utils.show(img)
    
    img1 = smoothImage(img)
    utils.show(img1)
    cv2.imwrite("result1.jpg", img1)
    
    img2 = binarizeImage(img, 125)
    utils.show(img2)
    cv2.imwrite("result2.jpg", img2)
   
    img3 = processImage(img)
    utils.show(img3)
    cv2.imwrite("result3.jpg", img3)
    
    img4 = doSomething(img)
    utils.show(img4)
    cv2.imwrite("result4.jpg", img4)
