from utils import *

def noisy(image):
    mu, sigma, n = np.mean(image), 30, image.shape[0]*image.shape[1]
    # adding gaussian noise
    gaussian_noise = np.random.normal(mu, sigma, n).reshape(image.shape[0],image.shape[1])
    return image + gaussian_noise


gaussian_input21_L = noisy(input21_left)
gaussian_input21_R = noisy(input21_right)
plt.imshow(gaussian_input21_L, cmap = 'gray')
plt.title("input 2 left with gaussian noise")
plt.show()



D2_l_to_r_gaussian = basic_stereo_algo(gaussian_input21_L, input21_right, 
                                       verbose = False, l_to_r= True)

D2_l_to_r_gaussian_both = basic_stereo_algo(gaussian_input21_L, gaussian_input21_R,
                                            verbose = False, l_to_r= True)

D2_r_to_l_gaussian_both = basic_stereo_algo(gaussian_input21_R, gaussian_input21_L,
                                            verbose = False, l_to_r= False)

plt.imshow(np.absolute(D2_l_to_r_gaussian), cmap = 'gray')
plt.title("Disparity l to r | input 2 | l gaussian noise")
plt.show()

plt.imshow(np.absolute(D2_l_to_r_gaussian_both), cmap = 'gray')
plt.title("Disparity l to r | input 2 | l and r gaussian noise")
plt.show()

plt.imshow(np.absolute(D2_r_to_l_gaussian_both), cmap = 'gray')
plt.title("Disparity r to l | input 2 | l and r gaussian noise")
plt.show()


misc.imsave('./output/ps2-3-a-1.png', np.absolute(D2_l_to_r_gaussian_both))
misc.imsave('./output/ps2-3-a-2.png', np.absolute(D2_r_to_l_gaussian_both))

def change_contrast(img, factor):
    img_bis = np.array(img, dtype = 'int32')
    contrasted = img_bis * factor
    contrasted[contrasted >= 255] = 255
    return contrasted


input21_left_contrast = change_contrast(input21_left, 1.1)
input21_right_contrast = change_contrast(input21_right, 1.1)

D2_l_to_r_contrast = basic_stereo_algo(input21_left_contrast, 
                                       input21_right, 
                                       verbose = False, l_to_r= True)

D2_l_to_r_contrast_both = basic_stereo_algo(input21_left_contrast, 
                                            input21_right_contrast, 
                                            verbose = False, l_to_r= True)


D2_r_to_l_contrast_both = basic_stereo_algo(input21_right_contrast, 
                                            input21_left_contrast, 
                                            verbose = False, l_to_r= False)


plt.imshow(np.absolute(D2_l_to_r_contrast), cmap = 'gray')
plt.title("Fig. 2b1) Disparity l to r | input 2 | l with constrat")
plt.show()



plt.imshow(np.absolute(D2_l_to_r_contrast_both), cmap = 'gray')
plt.title("Fig. 2b2) Disparity l to r | input 2 | l and r with constrat")
plt.show()


plt.imshow(np.absolute(D2_r_to_l_contrast_both), cmap = 'gray')
plt.title("Fig. 2b3) Disparity r to l | input 2 | l and r with constrat")
plt.show()


misc.imsave('./output/ps2-3-b-1.png', np.absolute(D2_l_to_r_contrast_both))
misc.imsave('./output/ps2-3-b-2.png', np.absolute(D2_r_to_l_contrast_both))




