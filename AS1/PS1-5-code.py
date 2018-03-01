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


def hough_transform_circle(image, image_edges, max_rad):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Initialize empty accumulator (filled with 0)
    height = image.shape[0]
    width = image.shape[1]

    hough_accu = np.zeros([height, width, max_rad])
    # get row and columns indexes for all indexes 
    rr_indexes, cc_indexes = np.nonzero(image_edges)
    
    # compute gradient xx and yy from original picture
    gxx = cv2.Sobel(image,cv2.CV_32FC1,1,0);
    gyy = cv2.Sobel(image,cv2.CV_32FC1,0,1);
    # compute gradient direction for each point
    theta_values = cv2.phase(gxx,gyy,angleInDegrees=True);

    # Browsing into each pixel of edges picture
    for k in range(len(rr_indexes)):
        # getting indexes of edge
        y = rr_indexes[k]
        x = cc_indexes[k]
        theta = np.deg2rad(theta_values[y,x])

        for radius in range(0, max_rad):
            a = x + radius * np.cos(theta)
            b = y + radius * np.sin(theta)
            if (b < height) & ( a>=0 ) & ( b>=0 ) & (a < width) :
                hough_accu[b, a, radius] += 1

    return hough_accu

def peak_finding_circle(hough_accu_circle, min_dist = 20, max_peaks=10, title ='Hough Transform circle'): 
    coordinates = peak_local_max(hough_accu_circle, min_distance = min_dist,
                                 exclude_border = False, num_peaks = max_peaks)
    loca_maxs_b = coordinates[:, 0]
    loca_maxs_a = coordinates[:, 1]
    loca_maxs_radius = coordinates[:, 2]
    
    return loca_maxs_a, loca_maxs_b, loca_maxs_radius


def draw_circle(image, loca_maxs_a, loca_maxs_b, loca_maxs_radius, min_rad = 0, title = 'Detected Circle'):
    image_copy = image.copy()
    for j in range(len(loca_maxs_a)):
        a = loca_maxs_a[j]
        b = loca_maxs_b[j]
        radius = loca_maxs_radius[j]
        if (radius > min_rad): 
            cv2.circle(image_copy, (a, b), radius, (255,0,0), 2)
        #print('Cercle {} | b(row)= {} a(col) = {} radius = {}'.format(j,loca_maxs_b[j], loca_maxs_a[j], loca_maxs_radius[j]))
    plt.imshow(image_copy)
    plt.title(title)

    plt.show()
    return image_copy

input1 = cv2.imread('./ps1-input1.jpg')
input1_gaussian = ndimage.filters.gaussian_filter(input1, sigma=3)
plt.imshow(input1_gaussian, cmap='gray', aspect= 'auto')
plt.show()

misc.imsave('./output/ps1-4-a-1.png', input1_gaussian)

input1_gaussian_edges = cv2.Canny(input1_gaussian,30,40)
plt.imshow(input1_gaussian_edges, cmap = 'gray', aspect= 'auto')
plt.show()
misc.imsave('./output/ps1-4-b-1.png', input1_gaussian_edges)

hough_accu_circle = hough_transform_circle(input1_gaussian, input1_gaussian_edges, 100)

loca_maxs_a, loca_maxs_b, loca_maxs_radius = peak_finding_circle(
    hough_accu_circle,  max_peaks = 15, title='Hough Transform of input1_gaussian')

ps1_5_a_2 = draw_circle(input1_gaussian, loca_maxs_a, loca_maxs_b, loca_maxs_radius, title = "Detected Circle on input1 gaussian")

misc.imsave('./output/ps1-5-a-1.png', input1_gaussian_edges)
misc.imsave('./output/ps1-5-a-2.png', ps1_5_a_2)

