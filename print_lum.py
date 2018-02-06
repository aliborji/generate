import numpy as np
import scipy.stats as stats

imgdir = 'cartoon'
output_root = './%s/results/lum'%imgdir
tags = ['train', 'train2', 'dcgan', 'wgan', 'vae']

train_stat = output_root + '/' + tags[0] +'_lum_skew.npz'

train_stat = np.load(train_stat)['arr_0']
output_name = open(output_root + '/train_ave.txt', 'w')
print 'average skewness of training images is %.4f \n' % train_stat.mean()
output_name.write('skewness: %.4f \n' % train_stat.mean())
output_name.close()

for tag in tags[1:]:
    output_name = open(output_root + '/' + tag + '_stat_test.txt', 'w')
    gen_stat = np.load(output_root + '/'  +tag + '_lum_skew.npz')['arr_0']
    statistic, pvalue = stats.ttest_ind(train_stat, gen_stat, equal_var=False)
    print 'average skewness of %s is %.4f, ' % (tag, gen_stat.mean())
    print 't-statistic: %.4f, p-value: %.4f\n' % (statistic, pvalue)
    output_name.write('average: %.4f, ' % gen_stat.mean())
    output_name.write('statistic: %.4f, pvalue: %.4f\n' % (statistic, pvalue))
    output_name.close()