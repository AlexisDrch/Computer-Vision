
# coding: utf-8

# # <p style="text-align: center;"> CS6476 | Computer Vision ps5 </p>
# <p style="text-align: center;">Alexis Durocher - MSCS student at Georgia Tech</p>
# <p style="text-align: center;">Spring 2018</p>
# 

# In[1]:

import cv2

from scipy import ndimage
from scipy import misc
import math as mt
import numpy as np
import math as mtugh
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from  scipy.optimize import minimize
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


# <i> In this project, the aim is to use and analyze the Lukas Kanade method to detect flows in sequences of images. </i>

# In[2]:

def image_pair(image1, image2):
    return np.concatenate([image1, image2], axis =1)


# In[3]:

kernel_x = np.array([[-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]])

kernel_y = np.array([[-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]])


# ### 1. Laplacian/Gaussian pyramids

# In[4]:

yos_1 = cv2.imread('./images/DataSeq1/yos_img_01.jpg', 0)
yos_2 = cv2.imread('./images/DataSeq1/yos_img_02.jpg', 0)
yos_3 = cv2.imread('./images/DataSeq1/yos_img_03.jpg', 0)
plt.imshow(yos_1)
plt.show()


# #### 1.1  Reduce

# In[5]:

def reduce(img, k):
    """
    @summary reduce /2 a version of a given image, k times
    @params img: source image
    """
    reduced = img.copy()
    for i in range(0, k):
        reduced = cv2.resize(reduced, (0,0), fx = 0.5, fy = 0.5)
    return reduced

def get_gaussian_pyramid(image, n_levels = 4):
    gaussian_pyramid = {}
    gaussian_pyramid[0] = image

    for level in range(0, n_levels-1):
        img_n = gaussian_pyramid[level]
        smoothed = cv2.GaussianBlur(img_n,(5,5),0)
        gaussian_pyramid[level+1] = reduce(smoothed, 1)
    return gaussian_pyramid


# In[6]:

gaussian_pyramid = get_gaussian_pyramid(yos_1, 4)

for level, image in gaussian_pyramid.items():
    plt.imshow(image, cmap ='gray')
    plt.savefig('./output/PS5-1-1-'+str(level)+'.png')
    plt.show()


# #### 1.2 Expand 

# In[7]:

def expand(img, size):
    expanded = cv2.resize(img, size)
    return expanded

def get_laplacian_pyramid(image, n_levels):
    gaussian_pyramid = get_gaussian_pyramid(image, n_levels)
    laplacian_pyramid = {}
    for key, image in gaussian_pyramid.items():
        if (key == n_levels -1):
            laplacian_pyramid[key] = image
        else:
            level_img = gaussian_pyramid[key]
            level_plus_1_img = gaussian_pyramid[int(key) + 1]
            h, w = level_img.shape
            expanded = expand(level_plus_1_img, (w,h)) 
            laplacian_pyramid[key] = level_img - expanded
    return laplacian_pyramid


# In[8]:

laplacian_pyramid = get_laplacian_pyramid(yos_1, 4)
for level, image in laplacian_pyramid.items():
    plt.imshow(image, cmap ='gray')
    plt.savefig('./output/PS5-1-2-'+str(level)+'.png')
    plt.show()
