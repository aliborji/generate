import matplotlib.pyplot as plt
import os
from myfunc import luminance_distribution

# path to training images should be the first element
imgRoots = ['./generate/train',
            './generate/dcgan',
            './generate/wgan',
            './generate/vae',
            './generate/draw']
tags = ['train', 'dcgan', 'wgan', 'vae', 'draw']

output_root = './imagenet/results/lum'
if not os.path.exists(output_root):
    os.mkdir(output_root)
fig = plt.figure()
for tag, img_root in zip(tags, imgRoots):
    h, b, s = luminance_distribution(img_root, output_root, tag )
    plt.plot(b[1:], h/h.sum(), linewidth=3.0, label=tag, alpha=0.7)
plt.legend(fontsize=12)
plt.xscale('log')
plt.ylabel('Probability density', fontsize=16)
plt.xlabel('Relative luminance', fontsize=16)
plt.show()