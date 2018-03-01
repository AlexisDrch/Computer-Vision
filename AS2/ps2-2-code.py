from utils import *

D2_l_to_r = basic_stereo_algo(input21_left, input21_right, verbose = False, l_to_r= True)
D2_r_to_l = basic_stereo_algo(input21_right, input21_left, verbose = False, l_to_r= False)


plt.imshow(np.absolute(D2_l_to_r), cmap = 'gray')
plt.title("Disparity l to r | input 2 | no noise")
plt.show()
plt.imshow(np.absolute(D2_r_to_l), cmap = 'gray')
plt.title("Disparity r to l | input 2 | no noise")
plt.show()

misc.imsave('./output/ps2-2-a-1.png', np.absolute(D2_l_to_r))
misc.imsave('./output/ps2-2-a-2.png', np.absolute(D2_r_to_l))


plt.imshow(input21_Disp_left, cmap ='gray')
plt.title("input 2 left ground truth")
plt.show()