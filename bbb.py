from myfunc import color_grad_mag
import matplotlib.pyplot as plt
import cv2

img_name = '0_341.png'

img = cv2.imread(img_name)
img = img / 255.0

gradimg = color_grad_mag(img)

plt.imsave('%s_grad.png'%img_name[:-4], gradimg, cmap='gray')