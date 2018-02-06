from __future__ import print_function
import numpy as np
import scipy.stats as stats
from myfunc import weibull_contrast_param

imgdir = 'cartoon'
output_root = './%s/results/contrast'%imgdir
tags = ['train', 'train2', 'dcgan', 'wgan', 'vae']

for im, sta in enumerate(['c', 's', 'kld']):
    print(sta)
    val = []
    tval = []
    pval = []
    train_stat = np.load(output_root + '/' + tags[0] + '_%s.npz'%sta)['arr_0']
    for it, tag in enumerate(tags):
        gen_stat = np.load(output_root + '/' + tag + '_%s.npz'%sta)['arr_0']
        statistic, pvalue = stats.ttest_ind(train_stat, gen_stat, equal_var=False)
        val.append(gen_stat.mean())
        tval.append(statistic)
        pval.append(pvalue)
    for v in val:
        print(' &%.4f'%v, end='')
    print('\n')
    for t in tval:
        print(' &%.4f'%t, end='')
    print('\n')
    for p in pval:
        print(' &%.4f '%p, end='')
    print('\n')