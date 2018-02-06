from myfunc import ave_img
import matplotlib.pyplot as plt

img = ave_img('./imagenet/vae')
img = img[:, :, ::-1]
plt.imsave('ave.png', img/255)