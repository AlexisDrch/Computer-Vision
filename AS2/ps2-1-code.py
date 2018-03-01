from utils import *

plt.imshow(input1_left, cmap="gray")
plt.show()

D1_l_to_r = basic_stereo_algo(input1_left, input1_right, l_to_r = True)
D1_r_to_l = basic_stereo_algo(input1_right, input1_left, l_to_r = False)

D1_l_to_r_norm = np.absolute(D1_l_to_r)
D1_r_to_l_norm = np.absolute(D1_r_to_l)
plt.imshow(D1_l_to_r_norm, cmap ='gray')
plt.show()
plt.imshow(D1_r_to_l_norm, cmap ='gray')
plt.show()

misc.imsave('./output/ps2-1-a-1.png', D1_l_to_r_norm)
misc.imsave('./output/ps2-1-a-2.png', D1_r_to_l_norm)


synt_input1_right = compute_translation(input1_left, D1_l_to_r)
synt_input1_left = compute_translation(input1_right, D1_r_to_l)

plt.imshow(synt_input1_right, cmap= 'gray')
plt.title("Right input1 computed from Left input1")
plt.show()
plt.imshow(synt_input1_left, cmap= 'gray')
plt.title("Left input1 computed from Right input1")
plt.show()
