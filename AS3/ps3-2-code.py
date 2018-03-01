import cv2

from scipy import ndimage
from scipy import misc
import numpy as np
import math as mtugh
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from  scipy.optimize import minimize

# ### 2. Fundamental Matrix Estimation

# In[56]:


def convert_homog_to_nonhomog(pt):
    non_homog = pt/pt[-1]
    return np.array(non_homog[:-1])

image_a = cv2.imread("./pic_a.jpg",0)
image_b = cv2.imread("./pic_b.jpg",0)
# Now, the goal is to estimate the matching between points in one picture to lines in an other picture. We will use the fundamental matrix F.
# 

# In[14]:

data_2D_a = pd.read_csv('./pts2d-pic_a.csv')
data_2D_a.columns = ['ua','va']
data_2D_b = pd.read_csv('./pts2d-pic_b.csv')
data_2D_b.columns = ['ub','vb']
df_data_3 = pd.concat([data_2D_a, data_2D_b], axis = 1)
df_data_3 = pd.DataFrame(df_data_3)
df_data_3.head(n=5)


# In[15]:

# compute their relative position in both camera system using M


# #### 2.1) LSE to compute F 

# In[16]:

# function to build the matrix X, such that Xf = 0. With f unknown : a flattened version of F.
def build_X(df_data):
    X = np.array([0,0,0,0,0,0,0,0,0])
    #print(X)
    for index, pt in df_data.iterrows():
        new_row = np.array([pt['ub'] * pt['ua'], pt['ub'] * pt['va'], pt['ub'],
                            pt['vb'] * pt['ua'], pt['vb'] * pt['va'], pt['vb'],
                            pt['ua'], pt['va'], 1,])
        X = np.vstack((X, new_row))
    return X[1:]


# In[26]:

X = build_X(df_data_3)


# In[31]:

def compute_F_SVD(X):
    u, s, vh = np.linalg.svd(X, full_matrices=False)
    F = np.array(vh[-1], ndmin=2)
    F.shape = (3,3)
    return - F

F_SVD = compute_F_SVD(X)
print(F_SVD)


# #### 2.2)  Ajust rank of F 3 -> 2

# In[34]:

u, s, vh = np.linalg.svd(F_SVD, full_matrices=True)
assert(np.allclose(F_SVD, np.dot(u * s, vh)))
# remove last smaller singular value
s[2] = 0
print(np.diag(s))
# reconstruc F'
F_bis = np.dot(u*s, vh)
print(F_bis)
print(F_SVD)


# #### 2.3) Compute corresponding epipolar lines

# Recall that epipolar line ${l_a}$ (respectively ${l_b}$) corresponding to point ${p_b}$ (respectively ${p_a}$) can be computed with :
# 
# ${l_b}$ = F${p_a}$
# 
# ${l_a}$ = ${F^T}$${p_b}$ 

# In[35]:

# compute epipolar lines
lb_lines = np.array([0,0,0])
la_lines = np.array([0,0,0])

for pta, ptb in zip(data_2D_a.values, data_2D_b.values):
    pta = np.append(pta, [1])
    ptb = np.append(ptb, [1])
    lb = np.dot(F_bis, pta)
    la = np.dot(F_bis, ptb)
    lb_lines = np.vstack((lb_lines, lb))
    la_lines = np.vstack((la_lines, la))

lb_lines = lb_lines[1:]
la_lines = la_lines[1:]


# To draw the epilolar lines from their homogenous coordinates, we first need to find out the coordinates of the left and right line border of the picture. Indeed, this will allow us to compute the cross product with our lines and thus, find two points from which to draw lines.
# 
# With h and w the picture's height and width and X the cross-product, Recall :
# 
# ${l_L}$ = ${(0, 0, 1)}$ X ${(0, h, 1)}$
# 
# ${l_R}$ = ${(w, h, 1)}$ X ${(w, 0, 1)}$

# In[38]:

# find lL and lR
h, w = image_a.shape
l_L = np.cross(np.array([0, 0, 1]), np.array([0, h, 1]))
l_R = np.cross(np.array([w, 0, 1]), np.array([w, h, 1]))
print(l_L, l_R)


# We can now, find two points on each right and left border of the picture b (same for a) by computing :
#     
# ${pb_L}$ = ${l_L}$ X ${l_b}$
# 
# ${pb_R}$ = ${l_R}$ X ${l_b}$

# In[39]:

pts_al_lines = np.array([0,0])
pts_ar_lines = np.array([0,0])
pts_bl_lines = np.array([0,0])
pts_br_lines = np.array([0,0])

for lb_line, la_line in zip(lb_lines, la_lines):
    pb_l = np.array(convert_homog_to_nonhomog(np.cross(lb_line, l_L)), dtype= np.int32)
    pb_r = np.array(convert_homog_to_nonhomog(np.cross(lb_line, l_R)), dtype= np.int32)
    pa_l = np.array(convert_homog_to_nonhomog(np.cross(la_line, l_L)), dtype= np.int32)
    pa_r = np.array(convert_homog_to_nonhomog(np.cross(la_line, l_R)), dtype= np.int32)
    pts_al_lines = np.vstack((pts_al_lines, pa_l))
    pts_ar_lines = np.vstack((pts_ar_lines, pa_r))
    pts_bl_lines = np.vstack((pts_bl_lines, pb_l))
    pts_br_lines = np.vstack((pts_br_lines, pb_r))

pts_al_lines = pts_al_lines[1:]
pts_ar_lines = pts_ar_lines[1:]
pts_bl_lines = pts_bl_lines[1:]
pts_br_lines = pts_br_lines[1:]



# draw line on picture a 
for pta_l, pta_r, ptb_l, ptb_r in zip(pts_al_lines, pts_ar_lines, pts_bl_lines, pts_br_lines):
    image_a_with_lines = cv2.line(image_a, tuple(pta_l), tuple(pta_r), (0,255,0), 2)
    image_b_with_lines = cv2.line(image_b, tuple(pta_l), tuple(pta_r), (0,255,0), 2)


plt.imshow(image_a_with_lines)
plt.imsave("./output/image_a_with_lines.png", image_a_with_lines)
plt.show()

plt.imshow(image_b_with_lines)
plt.imsave("./output/image_b_with_lines.png", image_b_with_lines)
plt.show()
