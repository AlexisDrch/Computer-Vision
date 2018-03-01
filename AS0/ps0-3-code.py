# ### 3. Replacement of pixels

from scipy import misc
from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# #### a.

# In[39]:


# input 2 pictures as numpy ndarray
picture_1 = misc.imread('./pictures/ps0-1-a-1.jpg')
picture_2 = misc.imread('./pictures/ps0-1-a-2.jpg')

# set red and blue channel to value 0
mono_g_picture = picture_1.copy()
mono_g_picture[:,:,0] = mono_g_picture[:,:,2] = 0

# We take the inner square of size 100x100 from the M1g (Green Monochrome of image 1)
inner_mg1 = mono_g_picture[205:305, 205:305,:].copy()

print('The size of the mg1 inner square is {} x {}'.format(inner_mg1.shape[0], inner_mg1.shape[1]))

# build M2g (Green Monochrome of image 2)
mono_g2_picture = picture_2.copy()
mono_g2_picture[:,:,0] = mono_g2_picture[:,:,2] = 0

# replace inner square with mg1's one
mono_g2_picture[205:305, 205:305,:] = inner_mg1

# plot (for notebook)
#plt.imshow(mono_g2_picture)
#plt.title('Inner square of mg1 in mg2')
#plt.show()

mpimg.imsave('./output/ps0-3-a-1.jpg', mono_g2_picture)

exit()