# ### 4. Arithmetic and Geometric operations
from scipy import misc
from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# ### a. 



# input 2 pictures as numpy ndarray
picture_1 = misc.imread('./pictures/ps0-1-a-1.jpg')
picture_2 = misc.imread('./pictures/ps0-1-a-2.jpg')

# set red and blue channel to value 0
mono_g_picture = picture_1.copy()
mono_g_picture[:,:,0] = mono_g_picture[:,:,2] = 0

# In[40]:

green_mg1_values = mono_g_picture[:,:,1].copy()
min_g1_value = np.min(green_mg1_values)
max_g1_value = np.max(green_mg1_values)
mean_g1_value = np.mean(green_mg1_values)
std_g1_value = np.std(green_mg1_values)

print('From the MG1 pixel values : min = {} | max = {} | mean = {} | stand dev = {} '
      .format(min_g1_value, max_g1_value, mean_g1_value, std_g1_value))
print('\n')
print('To compute these values, it is necessary to consider the pixel values as a unique array,' +
      'here : the green pixel value of all the instances in the picture (green channel). ' +
      'Then, basic mathematic ops can be applied.')


# #### b. Operations on mg1

# In[41]:

# substracting the mean
green_mg1_values = green_mg1_values - mean_g1_value
# diving by the std
green_mg1_values = green_mg1_values / std_g1_value
# multiply by 10
green_mg1_values = green_mg1_values * 10
# add mean
green_mg1_values = green_mg1_values + mean_g1_value

# plot (for notebook) and output the resulting picture
mono_g_picture_flat = mono_g_picture.copy()
mono_g_picture_flat[:,:,1] = green_mg1_values
#plt.imshow(mono_g_picture_flat)
#plt.title('Flat M1g')
#plt.show()

mpimg.imsave('./output/ps0-4-b-1.jpg', mono_g_picture_flat)


# #### c. Shift M1g

# In[42]:

shifted_mg1 = mono_g_picture.copy()

#shift two pixels to the left, except two last columns
for i in range(512):
    for j in range(510):
        shifted_mg1[i,j] = shifted_mg1[i, j+2]

# plot (for notebook) and output resulting picture
#plt.imshow(shifted_mg1)
#plt.show()

mpimg.imsave('./output/ps0-4-c-1.jpg', shifted_mg1)


# #### d. M1g - shiftedM1g

# In[47]:

sub_m1g = mono_g_picture - shifted_mg1

# verif that green chanel has valid values (not < 0)
verif_array = np.where(sub_m1g < 0)
print(verif_array)

# plot (for notebook) and output resulting picture
#plt.imshow(sub_m1g)
#plt.show()

mpimg.imsave('./output/ps0-4-d-1.jpg', sub_m1g)


# The value of a pixel represent its light intensity. Since negative light intensity doesn't exist, negative value for a pixel is a bug, and does not represent a physical quantity.
exit()