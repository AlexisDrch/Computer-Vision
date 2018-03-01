# ### 5. Noise
from scipy import misc
from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# a. Gaussian noise on green channel - picture 1


# input 2 pictures as numpy ndarray
picture_1 = misc.imread('./pictures/ps0-1-a-1.jpg')
picture_2 = misc.imread('./pictures/ps0-1-a-2.jpg')

# In[44]:

gaussnoised_picture = picture_1.copy()
green_pixs = gaussnoised_picture[:,:,1]
mu, sigma, n = np.mean(green_pixs), 100, np.size(green_pixs)

# adding gaussian noise
gaussian_noise = np.random.normal(mu, sigma, n).reshape(512,512)
gaussnoised_green_pixs = green_pixs + gaussian_noise

gaussnoised_picture[:,:,1] = gaussnoised_green_pixs

#plot (for notebook) and output resulting picture
#plt.imshow(gaussnoised_picture)
#plt.title('ps0-5-a-1.jpg')
#plt.show()

misc.imsave('./output/ps0-5-a-1.jpg', gaussnoised_picture)

# Sigma corresponds to the standard deviation parameter used by the gaussian filter. The higher its value, the more effective the noise is.
# 

# b. Gaussian noise on blue chanel - picture 1

# In[45]:

gaussnoised_picture = picture_1.copy()
blue_pixs = gaussnoised_picture[:,:,2]
mu, sigma, n = np.mean(blue_pixs), 100, np.size(blue_pixs)

gaussian_noise = np.random.normal(mu, sigma, n).reshape(512,512)
gaussnoised_blue_pixs = blue_pixs + gaussian_noise

gaussnoised_picture[:,:,2] = gaussnoised_blue_pixs

#plot (for notebook) and output resulting picture
#plt.imshow(gaussnoised_picture)
#plt.title('ps0-5-b-1.jpg')
#plt.show()

misc.imsave('./output/ps0-5-b-1.jpg', gaussnoised_picture)


# #### d. Noise rendering 
# In this case it all depends by what we define as better. A better noise is a noise that changes more the final rendering of a picture so it is different from the original one. In this case, the gaussian noise applied on the
# green chanel of picture 1 is 'better' : i.e it has a more visible effect.
# 
# If we consider the final 'look' as an 'objective beauty' criterian : The gaussian noise applied on the blue chanel gives a better 'beautiful' rendering since it doesnt change so much the balance of the different colors on the picture. It doesn't give this so much this 'error and noisy' effect.

exit()