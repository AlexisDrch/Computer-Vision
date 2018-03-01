# ### 1. <u> Input images <u>

# #### a.

# In[34]:

## Python dependencies
from scipy import misc
from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# input 2 pictures as numpy ndarray
picture_1 = misc.imread('./pictures/ps0-1-a-1.jpg')
picture_2 = misc.imread('./pictures/ps0-1-a-2.jpg')

# Make sure they are not larger than 512*512.
print('Picture 1 is of size : {} x {}'.format(str(picture_1.shape[0]), str(picture_1.shape[1])))
print('Picture 2 is of size : {} x {}'.format(str(picture_2.shape[0]), str(picture_2.shape[1])))

# Plot pictures (for notebook purpose only)
#fig, (p1, p2) = plt.subplots(2,1)
#p1.imshow(picture_1)
#p2.imshow(picture_2)
#plt.show()

# Output pictures in ./output folder
mpimg.imsave("./output/ps0-1-a-1.jpg", picture_1)
mpimg.imsave("./output/ps0-1-a-2.jpg", picture_2)
print("image(s) in output repo")
exit()