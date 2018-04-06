 
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


 #### 2. Lucas Kanade
# 
# Image U are shift values from image 1 to image 2 along x axis (shift in column values)
# 
# Image V are shift values from image 1 to image 2 along y axis (shigt in row values)

# In[9]:

shift0 = cv2.imread('./images/TestSeq/Shift0.png', 0)
shiftR2 = cv2.imread('./images/TestSeq/ShiftR2.png', 0)
shiftR5U5 = cv2.imread('./images/TestSeq/ShiftR5U5.png', 0)
shiftR10 = cv2.imread('./images/TestSeq/ShiftR10.png', 0)
shiftR20 = cv2.imread('./images/TestSeq/ShiftR20.png', 0)
shiftR40 = cv2.imread('./images/TestSeq/ShiftR40.png', 0)
shift0_smo = cv2.GaussianBlur(shift0, (3,3), 5)
shiftR2_smo = cv2.GaussianBlur(shiftR2, (3,3), 5)
shiftR5U5_smo = cv2.GaussianBlur(shiftR5U5, (3,3), 5)
shiftR10_smo = cv2.GaussianBlur(shiftR10, (3,3),5)
shiftR20_smo = cv2.GaussianBlur(shiftR20, (3,3), 5)
shiftR40_smo = cv2.GaussianBlur(shiftR40, (3,3), 5)


# In[10]:

def compute_sum(image, ks, r, c):
    sum_tot = image[r-ks:r+ks+1, c-ks:c+ks+1].sum()
    return sum_tot

def solve_lk(mat_grads, mat_t):
    a, b, c, d = mat_grads[0,0], mat_grads[0,1], mat_grads[1,0], mat_grads[1,1]
    e, f = mat_t[0], mat_t[1]
    v = (-f + (c*e)/a) / ( d - (b*c)/a)
    u = (-e - b*v) / a
    return u, v

def solve_lk_mat(mat_grads, mat_t, epsilon = 1e-10):
    vec = np.zeros((2,1))
    if np.linalg.det(mat_grads) > epsilon:
        vec = np.dot(np.linalg.inv(mat_grads), - mat_t)
    return vec[0], vec[1]


# In[11]:

def compute_lk(image1, image2, window_size = 10):
    ks = int((window_size-1)/2)
    It = cv2.subtract(image2, image1)
    Ix, Iy = np.gradient(image1)
    # weighted sum : gaussian kernel
    weights = cv2.getGaussianKernel(window_size, sigma = 3)
    tuple_w = (window_size, window_size)
    # matrix components
    
    Ixx = cv2.boxFilter(Ix * Ix, -1, tuple_w)
    Iyy = cv2.boxFilter(Iy * Iy, -1, tuple_w)
    Ixy = cv2.boxFilter(Ix * Iy, -1, tuple_w)
    Itx = cv2.boxFilter(It * Ix, -1, tuple_w)
    Ity = cv2.boxFilter(It * Iy, -1, tuple_w)
    
    image_u, image_v = np.zeros(image1.shape), np.zeros(image1.shape)
    height, width = image1.shape
    for r in range(ks, height-ks):
        for c in range(ks, width-ks):
            el1 = Ixx[r, c]
            el2 = Iyy[r, c]
            el3 = Ixy[r, c]
            el4 = Itx[r, c]
            el5 = Ity[r, c]
            mat_grads = np.array([
                [el1, el3],
                [el3, el2]
            ], np.float32)
            mat_t = np.array([el4, el5], np.float32)
            u, v = solve_lk_mat(mat_grads, mat_t)
            image_u[r, c] = u
            image_v[r, c] = v
    # use the max val to normalize
    return image_u, image_v


# In[12]:

def quiver_flow(img, flow_u, flow_v, step_fac = 50):
    x = np.arange(0, img.shape[1], 1)
    y = np.arange(0, img.shape[0], 1)
    x, y = np.meshgrid(x, y)
    plt.figure(figsize=(10,5))
    fig = plt.imshow(img, cmap = 'gray')
    # step to be display
    step = int(img.shape[0] / step_fac)
    plt.quiver(x[::step], y[::step],
               flow_u[::step], flow_v[::step], color = 'r',
               pivot = 'middle', headwidth=1, headlength =2)
    return plt


# #### 2.1 Lukas Kanade : shift 2 right 

# In[13]:

shiftR2_u, shiftR2_v = compute_lk(shift0_smo, shiftR2_smo, 10)


# In[14]:

plt = quiver_flow(shift0, shiftR2_u, shiftR2_v)
plt.title('Lukas Kanade flow - 2 right')
plt.show()


# #### Lukas Kanade : shift 2 right / 5 up

# In[15]:

shiftR5U5_U, shiftR5U5_V = compute_lk(shift0_smo, shiftR5U5_smo, 10)


# In[16]:

plt = quiver_flow(shift0, shiftR5U5_U, shiftR5U5_V)
plt.title('Lukas Kanade flow - 5 right 5 top')
plt.show()


# #### 2.2 Lukas Kanade : big shift 10, 20, 40

# In[17]:

shiftR10_U, shiftR10_V = compute_lk(shift0_smo, shiftR10_smo , 10)
shiftR20_U, shiftR20_V = compute_lk(shift0_smo, shiftR20_smo , 10)
shiftR40_U, shiftR40_V = compute_lk(shift0_smo, shiftR40_smo , 10)


# In[18]:

plt = quiver_flow(shift0, shiftR10_U, shiftR10_V)
plt.title('Lukas Kanade flow - 10 right')
plt.show()
plt = quiver_flow(shift0, shiftR20_U, shiftR20_V)
plt.title('Lukas Kanade flow - 20 right')
plt.show()
plt = quiver_flow(shift0, shiftR40_U, shiftR40_V)
plt.title('Lukas Kanade flow - 40 right')
plt.show()


# <u> Interpretation </u>
# Flows are not consistent because LK assumes (among other assumptions) a small shift in the pixel from one image to an other. Here, for pixels shift higher than 10, the assumptions are not verified anymore. Hence we can see that some quiver arrows are completely disturbed (i.e flow detections are wrong), especially at the right of the moving square.

# #### 2.3 Warp

# In[19]:

# return a backed warp image
def warp_back(img2, flow_u, flow_v):
    """
    @summary compute a warped back(img1) version of img2
    @params img2: the src image (shifted)
    @params flow_u: the shift along x axis, from img1 to img2
    @params flow_v: the shift along y axis, from img1 to img2
    """
    x = np.arange(0, img2.shape[1], 1) # width
    y = np.arange(0, img2.shape[0], 1) # height
    x, y = np.meshgrid(x, y)
    map_x = np.array(x + flow_u, dtype=np.float32)
    map_y = np.array(y + flow_v, dtype=np.float32)
    warpI2 = cv2.remap(img2, map_x, map_y, interpolation = cv2.INTER_LINEAR)
    # note : we need a second warping for NaN value due to linear interpolation
    warpI2bis = cv2.remap(img2, map_x, map_y, interpolation = cv2.INTER_NEAREST)
    warpI2[warpI2 == np.NAN] = warpI2bis[warpI2 == np.NaN]
    return warpI2


# In[20]:

warped_shiftR2 = warp_back(shiftR2, shiftR2_u,  shiftR2_v)
plt.imshow(warped_shiftR2, cmap = 'gray')
plt.show()


# <u> Interpretation </u>
# 
# The warped back image is aligning with shiftR0. Indeed this is a back warp so the output corresponds to 

# <u> Applied on DataSequence 1 </u>

# In[21]:

dataseq1_1 = cv2.imread('./images/DataSeq1/yos_img_01.jpg', 0);
dataseq1_2 = cv2.imread('./images/DataSeq1/yos_img_02.jpg', 0);
dataseq1_3 = cv2.imread('./images/DataSeq1/yos_img_03.jpg', 0);

dataseq1_2_flowu, dataseq1_2_flowv = compute_lk(dataseq1_1, dataseq1_2)
dataseq1_3_flowu, dataseq1_3_flowv = compute_lk(dataseq1_2, dataseq1_3)


# In[22]:

# plot flows
plt = quiver_flow(dataseq1_1, dataseq1_2_flowu, dataseq1_2_flowv)
plt.show()

plt = quiver_flow(dataseq1_2, dataseq1_3_flowu, dataseq1_3_flowv)
plt.show()


# In[23]:

# plot warped comparisons
dataseq1_1_warped = warp_back(dataseq1_2, dataseq1_2_flowu, dataseq1_2_flowv)
dataseq1_1_diff = cv2.subtract(dataseq1_1, dataseq1_2)
dataseq1_1_comp = np.vstack((dataseq1_1, dataseq1_1_warped, dataseq1_1_diff))

dataseq1_2_warped = warp_back(dataseq1_3, dataseq1_3_flowu, dataseq1_3_flowv)
dataseq1_2_diff = cv2.subtract(dataseq1_2, dataseq1_3)
dataseq1_2_comp = np.vstack((dataseq1_2, dataseq1_2_warped, dataseq1_2_diff))

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(1,2,1)
ax.imshow(dataseq1_1_comp, cmap ='gray')
bx = fig.add_subplot(1,2,2)
bx.imshow(dataseq1_1_comp, cmap ='gray')
plt.show()


# <u> Applied on DataSequence 2 </u>

# In[24]:

dataseq2_1 = cv2.imread('./images/DataSeq2/0.png', 0);
dataseq2_2 = cv2.imread('./images/DataSeq2/1.png', 0);
dataseq2_3 = cv2.imread('./images/DataSeq2/2.png', 0);

dataseq2_2_flowu, dataseq2_2_flowv = compute_lk(dataseq2_1, dataseq2_2)
dataseq2_3_flowu, dataseq2_3_flowv = compute_lk(dataseq2_1, dataseq2_3)


# In[25]:

# plot flows
plt = quiver_flow(dataseq2_1, dataseq2_2_flowu, dataseq2_2_flowv)
plt.show()

plt = quiver_flow(dataseq2_2, dataseq2_3_flowu, dataseq2_3_flowv)
plt.show()


# In[26]:

# plot warped comparisons
dataseq2_1_warped = warp_back(dataseq2_2, dataseq2_2_flowu, dataseq2_2_flowv)
dataseq2_1_diff = cv2.subtract(dataseq2_1, dataseq2_2)
dataseq2_1_comp = np.vstack((dataseq2_1, dataseq2_1_warped,dataseq2_1_diff))

dataseq2_2_warped = warp_back(dataseq2_3, dataseq2_3_flowu, dataseq2_3_flowv)
dataseq2_2_diff = cv2.subtract(dataseq2_2, dataseq2_3)
dataseq2_2_comp = np.vstack((dataseq2_2, dataseq2_2_warped, dataseq2_2_diff))


fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(1,2,1)
ax.imshow(dataseq2_1_comp, cmap ='gray')
bx = fig.add_subplot(1,2,2)
bx.imshow(dataseq2_2_comp, cmap ='gray')
plt.show()

