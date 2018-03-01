# ###  2. <u> Color planes <u>

# In[35]:

## Python dependencies
from scipy import misc
from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# input 2 pictures as numpy ndarray
picture_1 = misc.imread('./pictures/ps0-1-a-1.jpg')
picture_2 = misc.imread('./pictures/ps0-1-a-2.jpg')


# #### a. Swap red and blue pixels of image 1

# In[36]:

picture_temp = picture = picture_1.copy()

# swap red and blue pixels' values
picture[:,:,0] = picture_temp[:,:,2] # red pixel's value <- blue's one
picture[:,:,2] = picture_temp[:,:,0] # blue pixel's value <- red's one

# plot (for notebook)
#plt.imshow(picture)
#plt.title("Swap of red and blue pix on Picture 1")
#plt.show()

# Output pictures in ./output folder
mpimg.imsave("./output/ps0-2-a-1.jpg", picture)


# #### b. Monochrome image 1 (Green)

# In[37]:

# set red and blue channel to value 0
mono_g_picture = picture_1.copy()
mono_g_picture[:,:,0] = mono_g_picture[:,:,2] = 0

# plot (for notebook)
#plt.imshow(mono_g_picture)
#plt.title("Monochrome (M1g) version of picture 1")
#plt.show()

# Output pictures in ./output folder
mpimg.imsave("./output/ps0-2-b-1.jpg", mono_g_picture)


# #### c. Monochrome image 1 (Red)

# In[38]:

# set green and blue channel to value 0
mono_r_picture = picture_1.copy()
mono_r_picture[:,:,1] = mono_r_picture[:,:,2] = 0

# plot (for notebook)
#plt.imshow(mono_r_picture)
#plt.title("Monochrome (M1r) version of picture 1")
#plt.show()

# Output pictures in ./output folder
mpimg.imsave("./output/ps0-2-c-1.jpg", mono_r_picture)


# #### d. Red or Green ?

# The red-monochrome looks more homogenous in a monochrome sense as the red pixels 
# seem to be more omnipresents in the whole picture and have an average value high
# enought to not be considered as black. In the other hand, green-monochrome shows more
# heterogeneity, more difference between black and green. <br>
# Hence red-monochrome would be my choice regarding the 'best' monochrome picture and 
# the green-monochrom the choice for a computer vision algorithm seeking for clearer
# differences and shapes.

exit()