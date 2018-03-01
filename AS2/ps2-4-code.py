from utils import *

D31a_l_to_r = basic_stereo_algo(input21_left, 
                                input21_right, verbose = False,
                                l_to_r = True, norma_corr= True)

D31a_r_to_l = basic_stereo_algo(input21_right, 
                                input21_left, verbose = False, 
                                l_to_r = False, norma_corr= True)



D31a_l_to_r = np.absolute(D31a_l_to_r)
plt.imshow(D31a_l_to_r, cmap= 'gray')
plt.title("Fig. 4a1) Disparity norm-corr l to r | input 2 ")
plt.show()

D31a_r_to_l = np.absolute(D31a_r_to_l)
plt.imshow(D31a_r_to_l, cmap= 'gray')
plt.title("Fig. 4a2) Disparity norm-corr r to l | input 2 ")
plt.show()

misc.imsave('./output/ps2-4-a-1.png', np.absolute(D31a_l_to_r))
misc.imsave('./output/ps2-4-a-2.png', np.absolute(D31a_r_to_l))


# Contrast ? Norm corr doesnt care about pixels intensity : doesnt change . contrast or not
D4_l_to_r_contrast_both = basic_stereo_algo(
    input21_left_contrast, input21_right_contrast, 
    verbose = False, l_to_r= True, norma_corr= True)


D4_r_to_l_contrast_both = basic_stereo_algo(
    input21_right_contrast, input21_left_contrast, 
    verbose = False, l_to_r= False, norma_corr= True)

D4_l_to_r_contrast = basic_stereo_algo(
    input21_left_contrast, input21_right, 
    verbose = False, l_to_r= True, norma_corr= True)


plt.imshow(np.absolute(D4_l_to_r_contrast_both), cmap= 'gray')
plt.title("Fig. 4b11) Disparity norm-corr r to l | input 2 | l and r contrasted ")
plt.show()


plt.imshow(np.absolute(D4_r_to_l_contrast_both), cmap= 'gray')
plt.title("Fig. 4b12) Disparity norm-corr r to l | input 2 | l and r contrasted ")
plt.show()


plt.imshow(np.absolute(D4_l_to_r_contrast), cmap= 'gray')
plt.title("Fig. 4b13) Disparity norm-corr l to r | input 2 | l contrasted ")
plt.show()


misc.imsave('./output/ps2-4-b-1.png', np.absolute(D4_l_to_r_contrast_both))
misc.imsave('./output/ps2-4-b-2.png', np.absolute(D4_r_to_l_contrast_both))


# Gaussian ? Norm corr is robust to gaussian noise ? 
D4_l_to_r_gaussian = basic_stereo_algo(
    gaussian_input21_L, gaussian_input21_R, 
    verbose = False, l_to_r= True, norma_corr= True)


D4_r_to_l_gaussian = basic_stereo_algo(
    gaussian_input21_R, gaussian_input21_L,
    verbose = False, l_to_r= False, norma_corr= True)


plt.imshow(np.absolute(D4_l_to_r_gaussian), cmap= 'gray')
plt.title("Fig. 4b21) Disparity norm-corr l to r | input 2 | l and r gaussian ")
plt.show()


plt.imshow(np.absolute(D4_r_to_l_gaussian), cmap= 'gray')
plt.title("Fig. 4b22) Disparity norm-corr r to l | input 2 | l and r gaussian ")
plt.show()


misc.imsave('./output/ps2-4-b-3.png', np.absolute(D4_l_to_r_gaussian))
misc.imsave('./output/ps2-4-b-4.png', np.absolute(D4_r_to_l_gaussian))


