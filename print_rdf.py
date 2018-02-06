import numpy as np
import scipy.stats as stats

imgdir = 'imagenet'
output_root = './%s/results/rdf'%imgdir
tags = ['train', 'train2', 'dcgan', 'wgan', 'vae']


train_stat = [(output_root + '/' + tags[0] +'_filter1.npz'),  # \gamma in the paper
              (output_root + '/' + tags[0] +'_filter2.npz'),  # \beta in the paper
              (output_root + '/' + tags[0] +'_filter3.npz')]

output_name = open(output_root + '/train_ave.txt', 'w')

train_stat0 = np.load(train_stat[0])['arr_0']
train_stat1 = np.load(train_stat[1])['arr_0']
train_stat2 = np.load(train_stat[2])['arr_0']

output_name = open(output_root + '/train_ave.txt', 'w')
print 'average kurtosis of three filter responses of training images: %.4f, %.4f, %.4f\n' \
% (train_stat0.mean(), train_stat1.mean(), train_stat2.mean())
output_name.write('filter1: %.4f ' % train_stat0.mean())
output_name.write('filter2: %.4f ' % train_stat1.mean())
output_name.write('filter3: %.4f ' % train_stat2.mean())
output_name.close()

for tag in tags[1:]:
    output_name = open(output_root + '/' + tag + '_stat_test.txt', 'w')
    gen_stat = np.load(output_root + '/'  +tag + '_filter1.npz')['arr_0']
    statistic, pvalue = stats.ttest_ind(train_stat0, gen_stat, equal_var=False)
    print tag
    print 'kurtosis of filter1 response: %.4f, t-tatistics: %.4f, p-value: %.4f' \
    % (gen_stat.mean(), statistic, pvalue)
    output_name.write('filter1\n')
    output_name.write('average: %.4f, ' % gen_stat.mean())
    output_name.write('statistic: %.4f, pvalue: %.4f\n' % (statistic, pvalue))

    gen_stat = np.load(output_root + '/'  +tag + '_filter2.npz')['arr_0']
    statistic, pvalue = stats.ttest_ind(train_stat1, gen_stat, equal_var=False)
    print 'kurtosis of filter2 response: %.4f, t-tatistics: %.4f, p-value: %.4f' \
    % (gen_stat.mean(), statistic, pvalue)
    output_name.write('filter2\n')
    output_name.write('average: %.4f, ' % gen_stat.mean())
    output_name.write('statistic: %.4f, pvalue: %.4f\n' % (statistic, pvalue))

    gen_stat = np.load(output_root + '/'  +tag + '_filter3.npz')['arr_0']
    statistic, pvalue = stats.ttest_ind(train_stat2, gen_stat, equal_var=False)
    print 'kurtosis of filter3 response: %.4f, t-tatistics: %.4f, p-value: %.4f\n' \
    % ( gen_stat.mean(), statistic, pvalue)
    output_name.write('filter3\n')
    output_name.write('average: %.4f, ' % gen_stat.mean())
    output_name.write('statistic: %.4f, pvalue: %.4f\n' % (statistic, pvalue))

    output_name.close()