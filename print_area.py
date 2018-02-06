import numpy as np
import scipy.stats as stats

imgdir = 'cartoon'
output_root = './%s/results/area'%imgdir
tags = ['train', 'train2', 'dcgan', 'wgan', 'vae']


train_stat = [(output_root + '/' + tags[0] +'_param1.npz'),  # \gamma in the paper
              (output_root + '/' + tags[0] +'_param2.npz'),  # \beta in the paper
              (output_root + '/' + tags[0] +'_res.npz')]

output_name = open(output_root + '/train_ave.txt', 'w')

train_stat0 = np.load(train_stat[0])['arr_0']
train_stat1 = np.load(train_stat[1])['arr_0']
train_stat2 = np.load(train_stat[2])['arr_0']

output_name = open(output_root + '/train_ave.txt', 'w')
print 'param1, param2, res of training images: %.4f, %.4f, %.4f\n' \
% (train_stat0.mean(), train_stat1.mean(), train_stat2.mean())
output_name.write('param1: %.4f ' % train_stat0.mean())
output_name.write('param2: %.4f ' % train_stat1.mean())
output_name.write('res: %.4f ' % train_stat2.mean())
output_name.close()

for tag in tags[1:]:
    output_name = open(output_root + '/' + tag + '_stat_test.txt', 'w')
    gen_stat = np.load(output_root + '/'  +tag + '_param1.npz')['arr_0']
    statistic, pvalue = stats.ttest_ind(train_stat0, gen_stat, equal_var=False)
    print tag
    print 'param1: %.4f, t-tatistics: %.4f, p-value: %.4f' \
    % (gen_stat.mean(), statistic, pvalue)
    output_name.write('filter1\n')
    output_name.write('average: %.4f, ' % gen_stat.mean())
    output_name.write('statistic: %.4f, pvalue: %.4f\n' % (statistic, pvalue))

    gen_stat = np.load(output_root + '/'  +tag + '_param2.npz')['arr_0']
    statistic, pvalue = stats.ttest_ind(train_stat1, gen_stat, equal_var=False)
    print 'param2: %.4f, t-tatistics: %.4f, p-value: %.4f' \
    % (gen_stat.mean(), statistic, pvalue)
    output_name.write('filter2\n')
    output_name.write('average: %.4f, ' % gen_stat.mean())
    output_name.write('statistic: %.4f, pvalue: %.4f\n' % (statistic, pvalue))

    gen_stat = np.load(output_root + '/'  +tag + '_res.npz')['arr_0']
    statistic, pvalue = stats.ttest_ind(train_stat2, gen_stat, equal_var=False)
    print 'res: %.4f, t-tatistics: %.4f, p-value: %.4f\n' \
    % ( gen_stat.mean(), statistic, pvalue)
    output_name.write('filter3\n')
    output_name.write('average: %.4f, ' % gen_stat.mean())
    output_name.write('statistic: %.4f, pvalue: %.4f\n' % (statistic, pvalue))

    output_name.close()