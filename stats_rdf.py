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

imgRoots = ['./%s/train'%tag,
            './%s/train2'%tag,
            './%s/dcgan'%tag,
            './%s/wgan'%tag,
            './%s/vae'%tag]
tags = ['train', 'train2', 'dcgan', 'wgan', 'vae']
# imgRoots = ['./%s/train2'%tag]
# tags = ['train2']

output_root = './%s/results/rdf'%tag
if not os.path.exists(output_root):
    os.mkdir(output_root)

np.random.seed(1118)
kernel1 = np.random.uniform(size=(8, 8))
kernel1 = (kernel1 - kernel1.mean()) / np.sqrt(((kernel1 - kernel1.mean()) ** 2).sum())
np.random.seed(5)
kernel2 = np.random.uniform(size=(8, 8))
kernel2 = (kernel2 - kernel2.mean()) / np.sqrt(((kernel2 - kernel2.mean()) ** 2).sum())
np.random.seed(18)
kernel3 = np.random.uniform(size=(8, 8))
kernel3 = (kernel3 - kernel3.mean()) / np.sqrt(((kernel3 - kernel3.mean()) ** 2).sum())

for tag, img_root in zip(tags, imgRoots):
    fig = plt.figure()
    h, b, k = random_filter_response_distribution(img_root, kernel1, output_root, tag + '_filter1')
    plt.plot(b[1:], h / h.sum(), linewidth=3.0, label='Filter 1')

    h, b, k = random_filter_response_distribution(img_root, kernel2, output_root, tag + '_filter2')
    plt.plot(b[1:], h / h.sum(), linewidth=3.0, label='Filter 2')
    h, b, k = random_filter_response_distribution(img_root, kernel3, output_root, tag + '_filter3')
    plt.plot(b[1:], h / h.sum(), linewidth=3.0, label='Filter 3')
    plt.legend(fontsize=12)
    plt.yscale('log')
    plt.ylabel('Probability density', fontsize=16)
    plt.xlabel('Relative luminance', fontsize=16)
    # fig.savefig('%s/rdf_%s.pdf' % (output_root, tag))