from utils import *

D5a_l_to_r = basic_stereo_algo(
    input3_left, input3_right,
    verbose = False, l_to_r= True, norma_corr= False)

D5a_nocorr_l_to_r = basic_stereo_algo(
    input3_left, input3_right,
    verbose = False, l_to_r= True, norma_corr= True)


plt.imshow(np.absolute(ground_truth_3), cmap= 'gray')
plt.title("Fig. 5a1) Disparity ground truth l to r | input 3 ")
plt.show()


plt.imshow(np.absolute(D5a_l_to_r), cmap= 'gray')
plt.title("Fig. 5a2) Disparity SSD l to r | input 3")
plt.show()


plt.imshow(np.absolute(D5a_nocorr_l_to_r), cmap = 'gray')
plt.title("Fig. 5a3) Disparity norma corr l to r | input 3")
plt.show()


misc.imsave('./output/ps2-5-a-1.png', np.absolute(D5a_l_to_r))
misc.imsave('./output/ps2-5-a-2.png', np.absolute(D5a_nocorr_l_to_r))


D5a_nocorr_l_to_r_28 = basic_stereo_algo(
    input3_left, input3_right,
    verbose = False, l_to_r= True, norma_corr= True, window_size = 28)


plt.imshow(np.absolute(D5a_nocorr_l_to_r_28), cmap = 'gray')
plt.title("Fig. 5a4) Disparity norma corr l to r | input 3 - bigger window size")
plt.show()


D5a_nocorr_l_to_r_4 = basic_stereo_algo(
    input3_left, input3_right,
    verbose = False, l_to_r= True, norma_corr= True, window_size = 4)


plt.imshow(np.absolute(D5a_nocorr_l_to_r_4), cmap = 'gray')
plt.title("Fig. 5a5) Disparity norma corr l to r | input 3 - smaller window size")
plt.show()


D5a_nocorr_l_to_r_4_smoo = ndimage.filters.median_filter(D5a_nocorr_l_to_r_4, size= 8)


plt.imshow(np.absolute(D5a_nocorr_l_to_r_4_smoo), cmap = 'gray')
plt.title("Fig. 5a6) Disparity norma corr l to r | input 3 - smaller window size | median filter")
plt.show()


misc.imsave('./output/ps2-5-a-3.png', np.absolute(D5a_nocorr_l_to_r_28))
misc.imsave('./output/ps2-5-a-4.png', np.absolute(D5a_nocorr_l_to_r_4_smoo))


input3_left_smoo = ndimage.filters.gaussian_filter(input3_left, sigma=3)
input3_right_smoo = ndimage.filters.gaussian_filter(input3_right, sigma=3)


D5b_l_to_r_gauss = basic_stereo_algo(
    input3_left_smoo, input3_right_smoo,
    verbose = False, l_to_r= True, norma_corr= False)

D5b_nocorr_l_to_r_gauss = basic_stereo_algo(
    input3_left_smoo, input3_right_smoo,
    verbose = False, l_to_r= True, norma_corr= True)


plt.imshow(np.absolute(ground_truth_3), cmap= 'gray')
plt.title("Fig. 5b1) Disparity ground truth l to r | input 3")
plt.show()


plt.imshow(np.absolute(D5b_l_to_r_gauss), cmap= 'gray')
plt.title("Fig. 5b2) Disparity SSD l to r | input 3 smoothed")
plt.show()


plt.imshow(np.absolute(D5b_nocorr_l_to_r_gauss), cmap = 'gray')
plt.title("Fig. 5b3) Disparity norma corr l to r | input 3 smoothed")
plt.show()


misc.imsave('./output/ps2-5-b-1.png', np.absolute(D5b_l_to_r_gauss))
misc.imsave('./output/ps2-5-b-2.png', np.absolute(D5b_nocorr_l_to_r_gauss))


input3_left_cont = change_contrast(input3_left, 1.1)
input3_right_cont = change_contrast(input3_right, 1.1)


D5c_l_to_r = basic_stereo_algo(
    input3_left_cont, input3_right_cont,
    verbose = False, l_to_r= True, norma_corr= False)

D5c_nocorr_l_to_r = basic_stereo_algo(
    input3_left_cont, input3_right_cont,
    verbose = False, l_to_r= True, norma_corr= True)


D5c_l_to_r_small_window = basic_stereo_algo(
    input3_left_cont, input3_right_cont,
    verbose = False, l_to_r= True, norma_corr= False, window_size = 4)


plt.imshow(np.absolute(ground_truth_3), cmap= 'gray')
plt.title("Fig. 5b1) Disparity ground truth l to r | input 3")
plt.show()


plt.imshow(np.absolute(D5c_l_to_r), cmap= 'gray')
plt.title("Fig. 5b2) Disparity SSD l to r | input 3 contrasted")
plt.show()


plt.imshow(np.absolute(D5c_nocorr_l_to_r), cmap = 'gray')
plt.title("Fig. 5b3) Disparity norma corr l to r | input 3 contrasted")
plt.show()

plt.imshow(np.absolute(ndimage.filters.median_filter(D5c_l_to_r_small_window, size= 8)), cmap= 'gray')
plt.title("Fig. 5b4) Disparity SSD l to r | input 3 contrasted | small window | median filter")
plt.show()


misc.imsave('./output/ps2-5-c-1.png', np.absolute(D5c_l_to_r))
misc.imsave('./output/ps2-5-c-2.png', np.absolute(D5c_nocorr_l_to_r))
misc.imsave('./output/ps2-5-c-3.png', np.absolute(
    ndimage.filters.median_filter(D5c_l_to_r_small_window, size= 8)))