# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 19:38:36 2019

@author: lhfue
"""

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def pps_pipeline(org_image):
    """
    Creates the pre-processing pipeline used during training. 
    
    Parameters
    ----------
    org_image: original image ---->>> 16 bit numpy.
        The image will be introduced into the model before training.
    Returns
    -------
    std_image: standard image
        standard image after a series of processing
    """
    
    org_image = (org_image - np.min(org_image)) / (np.max(org_image) - np.min(org_image))
    org_image = np.uint8(org_image*255)
    
    std_image = org_image.copy()
    _,binary = cv2.threshold(org_image,100,1,cv2.THRESH_BINARY)  
    
    image,contours,hierarch=cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    areas = [cv2.contourArea(contour) for contour in contours]
    
    max_area_ind = areas.index(max(areas))
    mask = np.zeros_like(std_image)
    
    cv2.fillPoly(mask, [contours[max_area_ind]], 255)
    std_image = cv2.bitwise_and(std_image, std_image, mask = mask)
    
    std_image  = cv2.equalizeHist(std_image)
    return std_image


data_dir = 'datasets/CT/DM/valid' 
save_dir = 'datasets/CT/DM/valid_pps/'
image_name_list = os.listdir(data_dir)
image_list =[]


def ext_roi(image):
    image_orign = image.copy()
#    image = np.pad(image,((0,0),(5,5)),'constant',constant_values = (0, 255))
 #   print(image.shape)
    _,binary = cv2.threshold(image,100,1,cv2.THRESH_BINARY)  
    image,contours,hierarch=cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    areas = [cv2.contourArea(contour) for contour in contours]
    max_area_ind = areas.index(max(areas))
    mask = np.zeros_like(image_orign)
    cv2.fillPoly(mask, [contours[max_area_ind]], 255)
#    x, y, w, h = cv2.boundingRect(contours[max_area_ind])
#    mask = np.zeros_like(image_orign)
#    mask[y:y+h, x:x+w] = 255
    image_orign = cv2.bitwise_and(image_orign, image_orign, mask = mask)
    
    return image_orign

for i in range(len(image_name_list)):
    png_path = os.path.join(data_dir, image_name_list[i])
#    img = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.imread(png_path, 0)
#    img_clip = img.copy()
#    img_clip[img_clip<=200] = 200
#    img_clip[img_clip>=1000] = 1000
#   np.clip(img_clip, 0, 1500, out = img_clip)
#    img_clip = (img_clip - np.min(img_clip)) / (np.max(img_clip) - np.min(img_clip))
#    img = (img - np.min(img)) / (np.max(img) - np.min(img))
#    img = np.uint8(img*255)
#    img_clip = np.uint8(img_clip*255)
#    img_clip = ext_roi(img_clip)
#    img_clip = cv2.equalizeHist(img_clip)
    img_clip = pps_pipeline(img)
#    img_clip_blur = cv2.GaussianBlur(img_clip, (5,5,), 0)
    save_path = save_dir + image_name_list[i] 
    cv2.imwrite(save_path, img_clip)
    print("processing image: %d" % i)
#
#    plt.figure(figsize = (20,20))
#    plt.subplot(131)
#    plt.imshow(img, cmap = 'gray')
#    plt.title('orignal:%d' % i)
#    
#    plt.subplot(132)
#    plt.imshow(img_clip, cmap = 'gray')
#    plt.title('clip:%d' % i)
#    
#    plt.subplot(133)
#    plt.imshow(img_clip_blur, cmap = 'gray')
#    plt.title('clip_blur:%d' % i)
#    plt.show()
    
   