import matplotlib
matplotlib.use('agg')
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
from myfunc import area_statistic

tag = 'cartoon'

imgRoots = ['./%s/train'%tag,
            './%s/dcgan'%tag,
            './%s/wgan'%tag,
            './%s/vae'%tag]
tags = ['train', 'dcgan', 'wgan', 'vae']

output_root = './%s/results/area'%tag
if not os.path.exists(output_root):
    os.mkdir(output_root)
it = 1
for tag, img_root in zip(tags, imgRoots):
    fig = plt.figure()
    bins, param1, param2 = area_statistic(img_root, output_root, tag)
    m = param1.mean()
    b = param2.mean()
    s = np.arange(bins.size) + 1
    plt.plot(s, bins, '.')
    plt.plot(s, 10 ** b * s ** m, label='Linear fit')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(fontsize=18)
    fig.savefig('%s/area_%s.pdf' % (output_root, tag))
