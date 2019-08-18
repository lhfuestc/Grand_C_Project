# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 16:33:04 2019

@author: 64360
"""
import numpy as np
from scipy import misc
from matplotlib import pyplot
data_dir = 'datasets/CT/DM/train/CT_DM_10_0.npy'
image = np.load(data_dir)
pyplot.imshow(image, cmap = 'gray')

