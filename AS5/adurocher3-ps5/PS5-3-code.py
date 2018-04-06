 
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


 ### 3. Hierarchical K optic flow

# #### 3.1

# In[27]:

def compute_hiera_LK(input_L, input_R, n = 5, window_size = 5):
    level_k = n
    while level_k > 0:
        # reducing to level_k
        Lk = reduce(input_L, level_k-1)
        Rk = reduce(input_R, level_k-1)
        h,w = Lk.shape
        if level_k == n:
            U, V = np.zeros(Lk.shape), np.zeros(Lk.shape)
        else:
            U, V = 2*expand(U, (w,h)), 2*expand(V, (w,h))
        # warping left image with flow field in x and y
        Wk = warp_back(Lk, U, V)
        # computing lk between warped image Wk and Rk to compute new delta flow fields in x and y
        Dx, Dy = compute_lk(Wk, Rk, window_size)
        # update flow fields
        U, V = U + Dx, V + Dy
        # decrement k
        level_k -= 1
    return U, V, Wk



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

    
# <u> Shift Right 10 pixel </u>

# In[28]:

#δ pixels, you’ll need to set n to (at least) log2(δ).
flowU_R10, flowV_R10, warped_L_R10 = compute_hiera_LK(shift0_smo, shiftR10_smo, n = int(np.log2(10) +1))
plt = quiver_flow(shift0, flowU_R10, flowV_R10)
plt.show()
diff_R10 = cv2.subtract(shift0_smo, warped_L_R10)
plt.imshow(diff_R10, cmap='gray')
plt.title('Difference between 10-warped shift0, and original')
plt.show()


# <u> Shift R 20 pixels </u>

# In[29]:

#δ pixels, you’ll need to set n to (at least) log2(δ).
flowU_R20, flowV_R20, warped_L_R20 = compute_hiera_LK(shift0_smo, shiftR20_smo, n = int(np.log2(20) +1))
plt = quiver_flow(shift0, flowU_R20, flowV_R20)
plt.show()
diff_R20 = cv2.subtract(shift0_smo, warped_L_R20)
plt.imshow(diff_R10, cmap='gray')
plt.title('Difference between 20-warped shift0, and original')
plt.show()


# <u> Shift R 40 pixels </u>

# In[30]:

#δ pixels, you’ll need to set n to (at least) log2(δ).
flowU_R40, flowV_R40, warped_L_R40 = compute_hiera_LK(shift0_smo, shiftR40_smo, n = int(np.log2(40) +1))
plt = quiver_flow(shift0, flowU_R40, flowV_R40)
plt.show()
diff_R40 = cv2.subtract(shift0_smo, warped_L_R40)
plt.imshow(diff_R40, cmap='gray')
plt.title('Difference between 40-warped shift0, and original')
plt.show()


# ### Juggle Sequence

# In[31]:

juggle0 = cv2.imread('./images/Juggle/0.png', 0)
juggle1 = cv2.imread('./images/Juggle/1.png', 0)
juggle2 = cv2.imread('./images/Juggle/2.png', 0)
juggle0_smo = cv2.GaussianBlur(juggle0, (5,5), sigmaX = 5, sigmaY =5)
juggle1_smo = cv2.GaussianBlur(juggle1, (5,5), sigmaX = 5, sigmaY =5)
juggle2_smo = cv2.GaussianBlur(juggle2, (5,5), sigmaX = 5, sigmaY =5)


# In[32]:

#measuring max shift between image 1 and image 2 : shift of 35 pixels
flowU_juggle, flowV_juggle, warped_juggle = compute_hiera_LK(juggle1_smo, juggle2_smo, n = int(np.log2(35) +1))
plt = quiver_flow(juggle1_smo, flowU_juggle, flowV_juggle)
plt.show()
diff_juggle = cv2.subtract(juggle1_smo, warped_juggle)
plt.imshow(diff_juggle, cmap='gray')
plt.title('Difference between warped juggle1, and original')
plt.show()


# #### Lk between image I2 and I1

# ### Taxi Sequence

# In[33]:

temp_taxi = cv2.imread('./images/Taxis/taxi-00.jpg', 0)
h_taxis, w_taxis = temp_taxi.shape


# In[34]:

def compute_flows_segmentation(flow_u, flow_v, k = 4, scaled = True):
    """
    compute kmeans on x, y, flow_u and flow_v features
    @params flow_u: flow on x axis 
    @parmas flow_v: flow on y axis
    """
    #prepare spatial features (rows and cols)
    h, w = flow_u.shape
    cols = np.arange(0, w)
    rows = np.arange(0, h)
    cols, rows = np.meshgrid(cols, rows)
    cols = cols.flatten()
    rows = rows.flatten()
    # prepare flows' features (u and v)
    vals_u = flowU_taxi1.flatten()
    vals_v = flowV_taxi1.flatten()
    features_dataframe = pd.DataFrame(data = {
        'rows' : rows,
        'cols' : cols,
        'val_u': vals_u,
        'val_v': vals_v,
    }, columns = ['rows', 'cols', 'val_u', 'val_v'])
    features = features_dataframe.values
    if scaled:
        features = StandardScaler().fit_transform(features)
    kmeans = KMeans( n_clusters= k)
    kmeans.fit(features)
    segmented = np.reshape(kmeans.labels_, flow_u.shape)
    return segmented

def color_img_flow(img, segmented):
    """
    color images according to smaller segments (clusters' label)
    @params img: source image (gray)
    @parmas segmented: segments label
    """
    #color segments
    uniques = np.unique(segmented)
    min_clust = uniques[0]
    min_size = len(segmented.flatten())
    for cluster in uniques:
        size = len(segmented[segmented == cluster])
        if size < min_size:
            min_clust = cluster
            min_size = size
    img[segmented == min_clust] = (0,0,255)       

    return img


# In[35]:

sequence_length = 19
quivers = {}
segmented_flow = {}
colored_flow = cv2.cvtColor(temp_taxi, cv2.COLOR_GRAY2RGB)
for i in range(0, sequence_length):
    # get 00 to 18 and 19 taxi images
    str_i1 = ('0'+str(i))[-2:];
    str_i2 = ('0'+str(i+1))[-2:];
    taxi1 = cv2.imread('./images/Taxis/taxi-'+str_i1+'.jpg', 0)
    taxi2 = cv2.imread('./images/Taxis/taxi-'+str_i2+'.jpg', 0)
    # need smoothing ?
    flowU_taxi1, flowV_taxi1, warpe_taxi1 = compute_hiera_LK(taxi1, taxi2, n = int(np.log2(20) +1))
    quivers[i] = quiver_flow(taxi1, flowU_taxi1, flowV_taxi1)
    quivers[i].show()
    segments = compute_flows_segmentation(flowU_taxi1, flowV_taxi1, k = 4, scaled = True)
    colored_flow = color_img_flow(colored_flow, segments)


# <u> Explication </u>
# 
# For each pair of images in taxi sequence, we calculated the resulting LK flow.
# On top of this flow, we could evaluate clusters using ${<x, y, flow_U, flow_V>}$ as features for each pixel. Hence our clusters are based upon spatial information (gives spatial consistancy) and flow values.
# 
# Because we used spatial information, we had to choose a big enough k (here 4) to find a cluster which provides information on the flow value consistent with the spatial information. This cluster was the smalles in size (comparing to other big clusters which were just splitting the picture in continous shapes). So we could color the smallest cluster in blue. We reiterated over all the sequence to see the complete sequence's flow.
# 

# In[36]:

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(1,1,1)
ax.imshow(colored_flow)
plt.title('Flows segmented on Taxis sequence')
plt.show()


# Here, for each LK flows computed between a pai of Taxi data, we colored in blue the resuling smaller segment (kmeans). 
# We can see the flow of the moving cars : the white taxi will turn to its right right, the far left will go along the road from left to right and the far right as well, from right to left.
