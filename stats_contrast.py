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
from myfunc import contrast_distribution, my_weibull

tag = 'imagenet'
# imgRoots = ['./%s/train'%tag,
#             './%s/dcgan'%tag,
#             './%s/wgan'%tag,
#             './%s/vae'%tag]
# tags = ['train', 'dcgan', 'wgan', 'vae']
imgRoots = ['./%s/train2'%tag]
tags = ['train2']

output_root = './%s/results/contrast'%tag
if not os.path.exists(output_root):
    os.mkdir(output_root)
# fig, axes = plt.subplots(len(tags), 1, sharey=True)
# fig.add_subplot(111, frameon=False)
# plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
for tag, img_root in zip(tags, imgRoots):
    h, b = contrast_distribution(img_root)
    fig = plt.figure()
    plt.plot(b[1:], h/h.sum(), linewidth=3.0, label='Contrast distribution')
    popt, pcov = opt.curve_fit(my_weibull, np.arange(len(h))[1:], h[1:])  # throw away x=0!
    plt.plot(b[1:], my_weibull(np.arange(len(h)), *popt)/h.sum(), linewidth=3.0, linestyle='dashed', label='Weibull fit')
    plt.legend(fontsize=12)
    plt.ylabel('Probability density', fontsize=16)
    plt.xlabel('Contrast', fontsize=16)
    # fig.savefig('%s/contrast_%s.pdf'%(output_root, tag))