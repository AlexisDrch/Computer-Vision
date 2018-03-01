import cv2

from scipy import ndimage
from scipy import misc
import numpy as np
import math as mtugh
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.feature import peak_local_max
from skimage import io
from skimage.draw import circle, ellipse_perimeter

ps1_input0 = cv2.imread('./ps1-input0.png')
# use classic magnitude estimation and Sobel size = 7
ps1_input0_edges = cv2.Canny(ps1_input0,100,200,1,7,True) 

mpimg.imsave('./output/ps1-1-a-1.png', ps1_input0_edges, cmap="gray")

