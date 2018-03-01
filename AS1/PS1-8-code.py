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

input3 = cv2.imread('ps1-input3.jpg')
input3_gaussian = ndimage.filters.gaussian_filter(input3, sigma = 2)
input3_edges = cv2.Canny(input3_gaussian, 10, 60, 1, 3, True) 
plt.imshow(input3_edges, cmap = 'gray')
plt.show()

hough_accu_8a = hough_transform(input3_edges)

loca_maxs_rho_8, loca_maxs_theta_8 = peak_finding(hough_accu_8a, min_distance= 10, max_peaks=22)

input3_gaussian_with_line = draw_line(input3_gaussian, loca_maxs_rho_8, loca_maxs_theta_8)

hough_accu_8a_circle = hough_transform_circle(input3_gaussian, input3_edges, max_rad= 40)

loca_maxs_a_8, loca_maxs_b_8, loca_maxs_radius_8 = peak_finding_circle(hough_accu_8a_circle,
    min_dist = 20)

ps1_8_a_1 = draw_circle(input3_gaussian_with_line, loca_maxs_a_8, loca_maxs_b_8, loca_maxs_radius_8)

misc.imsave('./output/ps1-8-a-1.png', ps1_8_a_1)

def compute_ellipse_params(x1, y1, x2, y2):
    x0 = (x1 + x2)/2
    y0 = (y1 + y2)/2
    alpha = mt.atan((y2 - y1)/(x2 - x1))
    return int(x0), int(y0), int(alpha)

def compute_ellipse_minor_axis(a, d, f):
    theta = mt.acos(
        (mt.pow(a,2) + mt.pow(d,2) - mt.pow(f,2)) /
        (2 * a * d)
    )
    b = mt.sqrt(
        (mt.pow(a,2) * mt.pow(d,2) * mt.pow(mt.sin(theta),2)) / 
        (mt.pow(a,2) - mt.pow(d,2) * mt.pow(mt.cos(theta),2))
    )
    return int(b)


def hough_transform_ellispe(image, min_pair_dist = 5, min_treshold = 1):
    # empty 1D accumulator
    width = image.shape[1]
    height = image.shape[0]
    diag_len = np.ceil(np.sqrt(width * width + height * height)) # max distance for rho = length of diag
    ellipses = np.array([])
    hough_accu = np.zeros([diag_len])
    
    # get i and j indexes for all indexes 
    y_indexes, x_indexes = np.nonzero(image)
    
    
    # for each 1st pair (x1, y1)
    for x1 in x_indexes:
        for y1 in y_indexes:
            if (image[y1, x1] > 0):
                # for each 2nd pair (x2, y2):
                for x2 in x_indexes:
                    for y2 in y_indexes:
                        if (image[y2, x2] > 0):
                            if ((x2 != x1) & (y2 != y1)) :
                                # euclidian distance between pair1 and pair2
                                dist_p1_p2 = np.linalg.norm(
                                    np.array((x1, y1)) - np.array((x2, y2)))

                                if (dist_p1_p2 > min_pair_dist) :
                                    x0, y0, alpha = compute_ellipse_params(x1, y1, x2, y2)
                                    a = int(dist_p1_p2/2) # major axis a

                                    # for each 3rd pair (x, y):
                                    for x in x_indexes:
                                        for y in y_indexes:
                                            if (image[y, x] > 0):
                                                if (((x2 != x) & (y2 != y)) & 
                                                    ((x1 != x) & (y1 != y))):
                                                    # euclidian distance between pair1 and pair2
                                                    dist_p0_p3 = np.linalg.norm(
                                                        np.array((x0, y0)) - np.array((x, y)))
                                                    dist_p0_p1 = np.linalg.norm(
                                                        np.array((x0, y0)) - np.array((x1, y1)))
                                                    dist_p0_p2 = np.linalg.norm(
                                                        np.array((x0, y0)) - np.array((x2, y2)))
                                                    if (
                                                        (dist_p0_p3 > min_pair_dist) & 
                                                        ((dist_p0_p3 < dist_p0_p1) | 
                                                         (dist_p0_p3 < dist_p0_p2))):
                                                        d = dist_p0_p3
                                                        f = dist_p0_p2
                                                        b = compute_ellipse_minor_axis(a, d, f)
                                                        ## increment hough accu for this b
                                                        hough_accu[b] += 1

                                    # find max value in houg_accu
                                    max_b = np.argmax(hough_accu)
                                    votes_for_b = hough_accu[max_b]
                                    if (votes_for_b > min_treshold):
                                        major_axis = 2*a
                                        minor_axis = max_b
                                        ellipses = np.append(ellipses, 
                                                             [x0, 
                                                              y0, 
                                                              major_axis, 
                                                              minor_axis, 
                                                              alpha])
                                        # remove pixels of ellipse in the edges picture
                                        rr, cc = ellipse_perimeter(y0, 
                                                                   x0, 
                                                                   int(minor_axis/2), 
                                                                   int(major_axis/2), 
                                                                   orientation = alpha)
                                        y_indexes, x_indexes = np.nonzero(image)
                                        print(len(y_indexes))
                                        image[rr, cc] = 0
                                        y_indexes, x_indexes = np.nonzero(image)
                                        print(len(y_indexes))
                                        cv2.ellipse(input3_gaussian_cop,
                                                (ellipse[0], ellipse[1]),
                                                (ellipse[2], ellipse[3]),
                                                ellipse[4],0,360,255,1)
                                        plt.imshow(input3_gaussian_cop, cmap= 'gray')
                                        plt.show()
                                    hough_accu = np.zeros([diag_len])
                            
    return ellipses

