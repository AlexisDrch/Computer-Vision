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

def hough_transform(image):
    # Initialize empty accumulator (filled with 0)
    # 2 * diag length : for positive and negative max distance. 0 being diag_len
    # 180 because theta from 0 to 180
    width = image.shape[1]
    height = image.shape[0]

    diag_len = np.ceil(np.sqrt(width * width + height * height)) # max distance for rho = length of diag
    rho_range = int(2*diag_len)
    hough_accu = np.zeros([rho_range, 180])

    # get i and j indexes for all indexes 
    j_indexes, i_indexes = np.nonzero(image)

    # Browsing into each pixel of edges picture
    for k in range(len(j_indexes)):
        # getting indexes of edge
        i = i_indexes[k]
        j = j_indexes[k]

        # voting : for each value of theta
        for theta in range(0, 180):
            rho = int(np.round(i * np.cos(np.deg2rad(theta)) + j * np.sin(np.deg2rad(theta))) + diag_len)# positive index for rho
            hough_accu[rho, theta] += 1
    
    return hough_accu

# Peak finding
def peak_finding(hough_accu, min_distance = 45, max_peaks = 6, title ='Hough Transform', path = './trash.png'):

    coordinates = peak_local_max(hough_accu, min_distance=min_distance,
                                 exclude_border = False, num_peaks =max_peaks)
    loca_maxs_rho = coordinates[:, 0]
    loca_maxs_theta = coordinates[:, 1]
    plt.imshow(hough_accu, cmap='gray',aspect='auto')
    plt.title(title)

    # Annotate local maximum
    for i in range(len(loca_maxs_rho)):
        plt.annotate('X',xy=(loca_maxs_theta[i],loca_maxs_rho[i]), arrowprops=dict(facecolor='yellow', shrink=0.05),)
    plt.savefig(path)
    #plt.show()
    return loca_maxs_rho, loca_maxs_theta


def draw_line(image, loca_maxs_rho, loca_maxs_theta, rgb = (0,255,0)):
    image_copy = image.copy()
    width = image_copy.shape[1]
    height = image_copy.shape[0]
    diag_len = np.ceil(np.sqrt(width * width + height * height)) 
    for j in range(len(loca_maxs_rho)):
        rho = loca_maxs_rho[j] - diag_len
        theta = loca_maxs_theta[j]
        a = np.cos(np.deg2rad(theta))
        b = np.sin(np.deg2rad(theta))
        x1=int(a*rho - diag_len*b) 
        y1=int(b*rho + diag_len*a)
        x2=int(a*rho + diag_len*b)
        y2=int(b*rho - diag_len*a)
        #print(x1,y1,x2,y2)
        cv2.line(image_copy, (x1,y1),(x2,y2), rgb, 3) # green line
        #print('Line {} | rho = {} theta = {}'.format(j,loca_maxs_rho[j], loca_maxs_theta[j]))
        plt.imshow(image_copy)
        plt.title('Detected Line')

    #plt.show()
    return image_copy

    

input2 = cv2.imread('./ps1-input2.jpg')
input2_gaussian = ndimage.filters.gaussian_filter(input2, sigma = 5)
input2_edges = cv2.Canny(input2_gaussian, 40, 60, 1, 3, True) 
plt.imshow(input2_edges, cmap= 'gray')
plt.show()

hough_accu_6a = hough_transform(input2_edges)

loca_maxs_rho, loca_maxs_theta =  peak_finding(hough_accu_6a, 20, max_peaks = 15)

ps1_6_a_1 = draw_line(input2_gaussian, loca_maxs_rho, loca_maxs_theta)

misc.imsave('./output/ps1-6-a-1.png', ps1_6_a_1)

def filter_parallel_lines(loca_maxs_rho, loca_maxs_theta):
    new_loca_maxs_rho = np.array([])
    new_loca_maxs_theta = np.array([])
    for theta, rho in zip(loca_maxs_theta, loca_maxs_rho):
            # looking for parallel lines
            (indx,) = np.where((loca_maxs_theta == theta))
            if (indx.size > 1):
                new_loca_maxs_rho = np.append(new_loca_maxs_rho, rho)
                new_loca_maxs_theta = np.append(new_loca_maxs_theta, theta)

    return new_loca_maxs_rho, new_loca_maxs_theta

def filter_close_lines(loca_max_rho, loca_maxs_theta, max_dist):
    new_loca_maxs_rho = np.array([])
    new_loca_maxs_theta = np.array([])
    for theta, rho in zip(loca_maxs_theta, loca_maxs_rho):
            # looking for close line
            (indx,) = np.where((np.abs((loca_maxs_rho - rho)) < max_dist))
            if (indx.size > 1):
                new_loca_maxs_rho = np.append(new_loca_maxs_rho, rho)
                new_loca_maxs_theta = np.append(new_loca_maxs_theta, theta)

    return new_loca_maxs_rho, new_loca_maxs_theta


loca_maxs_rho, loca_maxs_theta = filter_parallel_lines(np.array(loca_maxs_rho), 
                                                     np.array(loca_maxs_theta))
loca_maxs_rho, loca_maxs_theta = filter_close_lines(np.array(loca_maxs_rho), 
                                                     np.array(loca_maxs_theta), 30)

ps1_6_c_1 = draw_line(input2_gaussian, loca_maxs_rho, loca_maxs_theta)

misc.imsave('./output/ps1-6-c-1.png', ps1_6_c_1)