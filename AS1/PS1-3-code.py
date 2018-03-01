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

ps1_input0_noised = cv2.imread('./ps1-input0-noise.png')
plt.imshow(ps1_input0_noised, cmap ='gray')
plt.show()

ps1_input0_gaussian = ndimage.filters.gaussian_filter(ps1_input0_noised, sigma=3)
plt.imshow(ps1_input0_gaussian, cmap = 'gray')
plt.show()
misc.imsave('./output/ps1-3-a-1.png', ps1_input0_gaussian)


# to avoid noise, we need to set a treshold high enough.
# But not too high so we keep all the edge lines 
ps1_input0_noised_edges = cv2.Canny(ps1_input0_noised,500,1100)
plt.imshow(ps1_input0_noised_edges, cmap = 'gray')
plt.title('Edge detected on ps1_input0_noise')
plt.show()
misc.imsave('./output/ps1-3-b-1.png', ps1_input0_noised_edges)
# easy to see the cross line ine the midle. But boundaries are hard to detect from noise.

# Now applying on the smoothed version
ps1_input0_gaussian_edges = cv2.Canny(ps1_input0_gaussian, 0, 80)
plt.imshow(ps1_input0_gaussian_edges, cmap='gray')
plt.title('Edge detected on ps1_input0_noise smoothed')
plt.show()
misc.imsave('./output/ps1-3-b-2.png', ps1_input0_gaussian_edges)
# no need to set a particular low_treshold; noise is already relatively nicely avoided.
# thanks to gaussian filter

hough_accu_noised = hough_transform(ps1_input0_noised_edges)
loca_maxs_rho, loca_maxs_theta = peak_finding(hough_accu_noised, max_peaks=7,
                                              title = 'Hough Transform of input0_noise')

ps1_3_c_4 = draw_line(ps1_input0_noised, loca_maxs_rho, loca_maxs_theta)


hough_accu_gaussian = hough_transform(ps1_input0_gaussian_edges)
loca_maxs_rho, loca_maxs_theta = peak_finding(hough_accu_gaussian,
                                              title='Hough Transform of input0_noise gaussian',
                                             path = './output/ps1_3_c_1.png') 
ps1_3_c_2 = draw_line(ps1_input0_gaussian, loca_maxs_rho, loca_maxs_theta)

misc.imsave('./output/ps1-3-c-2.png', ps1_3_c_2)
