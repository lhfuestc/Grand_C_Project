# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 18:31:14 2019

@author: 64360
"""

import os
import cv2
import numbers
import numpy as np
import matplotlib.pyplot as plt


def random_translation(seg_img, lab_img, probability=1.0, border_usage=0.5, default_border=0.25, label_of_interest=None, default_pixel=None, default_label=None):
    """
    Translates a random selection of the input by a random amount. The translation varies between datapoints, but is the same for
    all inputs of a single datapoint.

    Parameters
    ----------
    source : iterable
        An iterable over a number of datapoints where each datapoint is a tuple of a list of inputs and a parameter dictionary.
    probability: float
        The probability of translating the input. If it is below 1, some inputs will be passed through unchanged.
    border_usage: float
        If a label_of_interest is set, the border is defined as the area between the rectangular bounding box around the region of
        interest and the edge of the image. This parameter defines how much of that border may be used for translation or, in other
        words, how much of that border may end up outside of the new image.
    default_border: float
        The amount of translation that is possible with respect to the input size, if label_of_interest is either not specified or
        not found in the input. border_usage does not apply to the default_border.
    label_of_interest: int
        The label of interest in the ground truth.
    default_pixel: number or None
        The fill value for the image input. If None, the minimum value will be used.
    default_label: number or None
        The fill value for the ground truth input. If None, the minimum value will be used.

    Returns
    -------
    gen : generator
        A generator that yields each transformed datapoint as a tuple of a list of inputs and a parameter dictionary.
    """
    if not isinstance(probability, numbers.Number):
        raise TypeError("Probability must be a number! Received: {}".format(type(probability)))

    if not isinstance(border_usage, numbers.Number):
        raise TypeError("Border usage must be a number! Received: {}".format(type(border_usage)))

    if not isinstance(default_border, numbers.Number):
        raise TypeError("Default border must be a number! Received: {}".format(type(default_border)))

    if label_of_interest is not None and not isinstance(label_of_interest, numbers.Number):
        raise TypeError("Label of interest must be a number! Received: {}".format(type(label_of_interest)))

    if default_pixel is not None and not isinstance(default_pixel, numbers.Number):
        raise TypeError("Default pixel must be a number! Received: {}".format(type(default_pixel)))

    if default_label is not None and not isinstance(default_label, numbers.Number):
        raise TypeError("Default label must be a number! Received: {}".format(type(default_label)))

    if probability > 1.0 or probability < 0.0:
        raise ValueError("Probability must be between 0.0 and 1.0! Received: {}".format(probability))

    if border_usage > 1.0 or border_usage < 0.0:
        raise ValueError("Border usage must be between 0.0 and 1.0! Received: {}".format(border_usage))

    if default_border > 1.0 or default_border < 0.0:
        raise ValueError("Default border must be between 0.0 and 1.0! Received: {}".format(default_border))

    def non(s):
        return s if s < 0 else None

    def mom(s):
        return max(0, s)


    if(np.random.rand() < probability):

        if label_of_interest is None or label_of_interest not in lab_img:
            xdist = default_border * lab_img.shape[0]
            ydist = default_border * lab_img.shape[1]

        else:
            itemindex = np.where(lab_img == 255)

            xdist = min(np.min(itemindex[0]), lab_img.shape[0] - np.max(itemindex[0])) * border_usage
            ydist = min(np.min(itemindex[1]), lab_img.shape[1] - np.max(itemindex[1])) * border_usage

        ox = np.random.randint(-xdist, xdist) if xdist >= 1 else 0
        oy = np.random.randint(-ydist, ydist) if ydist >= 1 else 0

        fill_value = default_pixel if default_pixel is not None else np.min(seg_img)
        shift_seg = np.full_like(seg_img, fill_value)
        shift_seg[mom(ox):non(ox), mom(oy):non(oy), ...] = seg_img[mom(-ox):non(-ox), mom(-oy):non(-oy), ...]

        fill_value = default_label if default_label is not None else np.min(lab_img)
        shift_lab = np.full_like(lab_img, fill_value)
        shift_lab[mom(ox):non(ox), mom(oy):non(oy), ...] = lab_img[mom(-ox):non(-ox), mom(-oy):non(-oy), ...]

    return shift_seg, shift_lab



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


    image, label = random_translation(img, lab, probability=1.0, border_usage=0.8, default_border=0.25, label_of_interest=255, default_pixel=None, default_label=None)
    
    
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



