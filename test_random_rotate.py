# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 18:31:14 2019

@author: 64360
"""

import os
import cv2
import numbers
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt


def random_rotation(seg_img, lab_img, probability=1.0, upper_bound=180):
    """
    Rotates a random selection of the input by a random amount. The rotation varies between datapoints, but is the same for
    all inputs of a single datapoint.

    Parameters
    ----------
    source : iterable
        An iterable over a number of datapoints where each datapoint is a tuple of a list of inputs and a parameter dictionary.
    probability: float
        The probability of rotating the input. If it is below 1, some inputs will be passed through unchanged.
    upper_bound: number
        The maximum rotation in degrees.

    Returns
    -------
    gen : generator
        A generator that yields each transformed datapoint as a tuple of a list of inputs and a parameter dictionary.
    """
    if not isinstance(probability, numbers.Number):
        raise TypeError("Probability must be a number! Received: {}".format(type(probability)))

    if not isinstance(upper_bound, numbers.Number):
        raise TypeError("Upper bound must be a number! Received: {}".format(type(upper_bound)))

    if upper_bound < 0:
        raise ValueError("Upper bound must be greater than 0! Received: {}".format(upper_bound))
    elif upper_bound > 180:
        upper_bound = 180


    if(np.random.rand() < probability):

        angle = np.random.randint(-upper_bound, upper_bound)
        angle = (360 + angle) % 360
        print(angle)
        seg_img = scipy.ndimage.interpolation.rotate(seg_img, angle, reshape=False, order=1, cval=np.min(seg_img), prefilter=False)  # order = 1 => biliniear interpolation
        lab_img = scipy.ndimage.interpolation.rotate(lab_img, angle, reshape=False, order=0, cval=np.min(lab_img), prefilter=False)  # order = 0 => nearest neighbour
    return seg_img, lab_img


seg_img_dir = "datasets/CT/DM/test/"
lab_img_dir = "datasets/CT/GT/test/"


label_list = []
image_list = []


img_name_list = os.listdir(seg_img_dir)
lab_name_list = os.listdir(lab_img_dir)


for i in range(len(img_name_list)):
    img_path = seg_img_dir + img_name_list[i]
    lab_path = lab_img_dir + lab_name_list[i]
    
    
    img = cv2.imread(img_path, 0)
    lab = cv2.imread(lab_path, 0)


    image, label = random_rotation(img, lab, 9.0, 180)
    
    
    plt.figure(figsize = (20, 20))
    
    plt.subplot(141)
    plt.imshow(img, cmap = 'gray')
    plt.title("OrgImg: %d" % i)
    
    plt.subplot(142)
    plt.imshow(lab, cmap = 'gray')
    plt.title("OrgLab: %d" % i)
    
    plt.subplot(143)
    plt.imshow(image, cmap = 'gray')
    plt.title("RdmRotImg: %d" % i)
    
    plt.subplot(144)
    plt.imshow(label, cmap = 'gray')
    plt.title("RdmRotLab: %d" % i)
    
    plt.show()



