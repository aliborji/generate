import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os.path as osp
from myfunc import *
import os
import scipy.optimize as opt
import scipy.stats as statsimport
import matplotlib.pyplot as plt
import numpy as np
import os.path as osp
import os
import scipy.optimize as opt
import scipy.stats as stats
from myfunc import average_power_spectrum

tag = 'imagenet'

# imgRoots = ['./%s/train'%tag,
#             './%s/dcgan'%tag,
#             './%s/wgan'%tag,
#             './%s/vae'%tag]
# tags = ['train', 'dcgan', 'wgan', 'vae']
imgRoots = ['./%s/train2'%tag]
tags = ['train2']
output_root = './%s/results/spec'%tag
if not os.path.exists(output_root):
    os.mkdir(output_root)
it = 1
for tag, img_root in zip(tags, imgRoots):
    fig = plt.figure()
    f_img = average_power_spectrum(img_root, output_root, tag)
    img_size = 128

    # sb = np.fft.fftshift(f_img)
    # sbh = sb.mean(0)
    # sbh = sbh[img_size/2+1:]
    # sb = sb.mean(1)
    # sb = sb[img_size/2+1:]
    # f = np.arange(img_size/2)[1:]
    # plt.plot(np.log(f), np.log10(sbh), label='Horizontal')
    # plt.plot(np.log(f), np.log10(sb), label='Vertical')
    # plt.legend(fontsize=12)
    # plt.xlabel('Log frequency')
    # plt.ylabel('Log power magnitude')
    # plt.grid()
    """surface plot"""
    ax = fig.add_subplot(111, projection='3d')
    x = np.arange(-img_size / 2, img_size / 2, 1)
    y = np.arange(-img_size / 2, img_size / 2, 1)
    x, y = np.meshgrid(x, y)
    ax.plot_surface(x, y,  np.log10(np.fft.fftshift(f_img)), rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))
    ax.set_xlabel('$f_x$', fontsize=18)
    ax.set_ylabel('$f_y$', fontsize=18)
    handles, labels = ax.get_legend_handles_labels()
    ax.grid('on')
    # fig.savefig('%s/spec_%s.pdf' % (output_root, tag))