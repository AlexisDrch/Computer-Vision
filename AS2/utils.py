import cv2

from scipy import ndimage
from scipy import misc
import numpy as np
import math as mtugh
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.feature import peak_local_max
from skimage import io

input1_left = cv2.imread("./Data/leftTest.png", 0)
input1_right = cv2.imread("./Data/rightTest.png", 0)

input21_left = cv2.imread("./Data/proj2-pair1-L.png", 0)
input21_right = cv2.imread("./Data/proj2-pair1-R.png", 0)

input21_Disp_left = cv2.imread('./Data/proj2-pair1-Disp-L.png')
input21_Disp_right = cv2.imread('./Data/proj2-pair1-Disp-R.png')


input3_left = cv2.imread("data/proj2-pair2-L.png", 0)
input3_right = cv2.imread("data/proj2-pair2-R.png", 0)
ground_truth_3 = cv2.imread("data/proj2-pair2-Disp-L.png", 0)

def compute_SSD(window0, window1):
    # cast to avod
    window_sub = window0 - window1
    window_ssd = np.square(window_sub)
    sum_ssd = np.sum(window_ssd)

    return sum_ssd

def compute_NORMA_CORR(row_dest, template):
    # cast to avoid
    res = cv2.matchTemplate(row_dest, template, cv2.TM_CCOEFF_NORMED)
    max_sim = np.argmax(res)
    return max_sim

def compute_norm_corr_on_dest_row(r0, c0, window0, img_dest, window_size = 10, l_to_r=True,
                                 max_offset = 110, min_offset = 30):
    height_dest, width_dest = img_dest.shape
    best_offset = 0
    min_disp = 100000000

    max_offset = max_offset
    min_offset = min_offset
    if l_to_r :
        tmp = max_offset
        max_offset = min_offset
        min_offset = tmp
    max_offset = int(min(c0 + max_offset + window_size / 2, width_dest -1))
    min_offset = int(max(c0 - min_offset - window_size /2, 0))

    top_row = int(max(r0 - window_size / 2, 0))
    down_row = int(min(r0 + window_size / 2, height_dest - 1))


    row_dest = np.array(img_dest[top_row: down_row, min_offset: max_offset] , dtype=np.float32)
    best_similarity = compute_NORMA_CORR(row_dest, window0)
    best_similarity = min_offset + best_similarity # back in col coordinates
    best_shift =  best_similarity - c0 + window_size/2

    return int(best_shift)

def compute_min_disp_on_dest_row(r0, c0, window0, img_dest, window_size=10, l_to_r=False,
                                max_offset = 110, min_offset = 30):
    height_dest, width_dest = img_dest.shape
    max_offset = max_offset
    min_offset = -min_offset
    best_offset = 0
    min_disp = 100000000

    for offset in range(min_offset, max_offset):
        if l_to_r:
            offset = -offset
        c = c0 + offset

        le_col = int(c - window_size / 2)
        ri_col = int(c + window_size / 2)
        if ((0 <= le_col) & (ri_col < width_dest)):
            top_row = int(r0 - window_size / 2)
            down_row = int(r0 + window_size / 2)
            window1 = np.array(img_dest[top_row: down_row, le_col: ri_col], dtype='int32')
            disp = compute_SSD(window0, window1)
            # plt.title("Right input1 computed from Left input1")
            plt.show()
            if disp < min_disp:
                min_disp = disp
                best_offset = offset
    # return best match (lowest ssd) 
    return best_offset


def basic_stereo_algo(img_ori, img_dest, verbose=False, l_to_r=False, norma_corr = False,
                     window_size = 16, max_offset = 110, min_offset = 30):
    height_ori, width_ori = img_ori.shape
    D = np.zeros(img_ori.shape)
    window_size = window_size
    # for each origin pixels
    for r in range(0, height_ori):
        if (verbose):
            print("--- " + str((r + 1) * 100 / height_ori) + " %")
        top_row = int(r - window_size / 2)
        down_row = int(r + window_size / 2)
        if ((0 <= top_row) & (down_row < height_ori)):
            for c in range(0, width_ori):
                le_col = int(c - window_size / 2)
                ri_col = int(c + window_size / 2)
                if ((0 <= le_col) & (ri_col < width_ori)):
                    # 1. fix an origin window centered in r, c
                    window_ori = np.array(
                        img_ori[top_row: down_row, le_col: ri_col], dtype=np.float32)
                    # 2 .fix a destination window centered in in r, c
                    # run the window on the horizontal row r of dest
                    if norma_corr :
                        min_ssd = compute_norm_corr_on_dest_row(r, 
                                                                c, 
                                                                window_ori, 
                                                                img_dest, 
                                                                window_size, 
                                                                l_to_r=l_to_r,
                                                                max_offset = max_offset,
                                                                min_offset = min_offset)
                    else :
                        min_ssd = compute_min_disp_on_dest_row(r,
                                                               c,
                                                               window_ori,
                                                               img_dest,
                                                               window_size,
                                                               l_to_r=l_to_r,
                                                               max_offset = max_offset,
                                                               min_offset = min_offset)
                    D[r, c] = min_ssd

    return D


def normalize_disparity(x):
    height, width = x.shape
    x_temp = np.zeros(x.shape)
    for i in range(0, height):
        for j in range(0, width):
            x_temp[i, j] = max(0, min(255, x[i, j]))
    return x_temp



def compute_translation(img_ori, D) :

    img_dest_synt = np.zeros(img_ori.shape)
    height, width = img_dest_synt.shape

    for i in range(0, height):
        for j in range(0, width):
            img_dest_synt[i,j] = img_ori[int(i + D[i,j]), j]
    return img_dest_synt





def noisy(image):
    mu, sigma, n = np.mean(image), 30, image.shape[0]*image.shape[1]
    # adding gaussian noise
    gaussian_noise = np.random.normal(mu, sigma, n).reshape(image.shape[0],image.shape[1])
    return image + gaussian_noise


def change_contrast(img, factor):
    img_bis = np.array(img, dtype = 'int32')
    contrasted = img_bis * factor
    contrasted[contrasted >= 255] = 255
    return contrasted







