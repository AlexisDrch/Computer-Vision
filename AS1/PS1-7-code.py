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


input2 = cv2.imread('./ps1-input2.jpg')
input2_gaussian = ndimage.filters.gaussian_filter(input2, sigma = 2)
input2_edges = cv2.Canny(input2_gaussian, 13000, 19000, 1, 7, True) 
plt.imshow(input2_edges, cmap= 'gray')
plt.show()

hough_accu_6a_circle = hough_transform_circle(input2_gaussian, input2_edges, max_rad= 100)

loca_maxs_a, loca_maxs_b, loca_maxs_radius =  peak_finding_circle(hough_accu_6a_circle, 
                                                                  max_peaks = 30,
                                                                  title='Hough Transform of input2 smoothed')

ps1_6_c_0 = draw_circle(input2_gaussian, loca_maxs_a, loca_maxs_b, loca_maxs_radius)

hough_accu_6a_circle = hough_transform_circle(input2_gaussian, input2_edges, max_rad= 40)

loca_maxs_a, loca_maxs_b, loca_maxs_radius =  peak_finding_circle(hough_accu_6a_circle, 
                                                                  max_peaks = 30,
                                                                  min_dist = 10,
                                                                  title='Hough Transform of input2 smoothed')

ps1_6_c_1 = draw_circle(input2_gaussian, loca_maxs_a, loca_maxs_b, loca_maxs_radius, min_rad = 20)

kernel = np.ones((5,5), np.uint8)
# applying erosion to flat the color to detect coins with similar color.
input2_li_erode = cv2.erode(input2_gaussian, kernel, iterations = 2)
plt.imshow(input2_li_erode)
plt.show()


def filter_homogenous_color_circle(img, loca_maxs_a, loca_maxs_b, loca_maxs_radius, std_treshold):
    new_as, new_bs, new_rads = np.array([], dtype = int), np.array([], dtype = int),np.array([], dtype = int)
    for a, b, radius in zip(loca_maxs_a, loca_maxs_b, loca_maxs_radius):
        rr, cc = circle(b, a, radius)
        stds = np.std(img[rr,cc])
        if (stds < std_treshold):
            new_as = np.append(new_as, int(a))
            new_bs = np.append(new_bs, int(b))
            new_rads = np.append(new_rads, int(radius))
    return new_as, new_bs, new_rads


loca_maxs_a_ho, loca_maxs_b_ho, loca_maxs_radius_ho = filter_homogenous_color_circle(input2_li_erode,loca_maxs_a, loca_maxs_b, loca_maxs_radius, std_treshold = 30)

ps1_7_a_1 = draw_circle(input2_gaussian, loca_maxs_a_ho, loca_maxs_b_ho, loca_maxs_radius_ho, min_rad = 20)

misc.imsave('./output/ps1-7-a-1.png', ps1_7_a_1)