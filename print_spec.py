from __future__ import print_function
import numpy as np
import scipy.stats as stats

imgdir = 'imagenet'
output_root = './%s/results/spec'%imgdir
tags = ['train', 'train2', 'dcgan', 'wgan', 'vae']

train_stat = [output_root + '/' + tags[0] +'_alpha_x.npz',  # \gamma in the paper
              output_root + '/' + tags[0] +'_alpha_y.npz',  # \beta in the paper
              output_root + '/' + tags[0] +'_a_x.npz',
              output_root + '/' + tags[0] +'_a_y.npz',
              output_root + '/' + tags[0] +'_res_x.npz',
              output_root + '/' + tags[0] +'_res_y.npz']

train_stat0 = np.load(train_stat[0])['arr_0']
train_stat1 = np.load(train_stat[1])['arr_0']
train_stat2 = np.load(train_stat[2])['arr_0']
train_stat3 = np.load(train_stat[3])['arr_0']
train_stat4 = np.load(train_stat[4])['arr_0']
train_stat5 = np.load(train_stat[5])['arr_0']

for im, sta in enumerate(['alpha_x', 'alpha_y', 'a_x', 'a_y', 'res_x', 'res_y']):
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


# output_name = open(output_root + '/train_ave.txt', 'w')
# print 'training images'
# print 'alpha_x: %.4f' % train_stat0.mean()
# print 'alpha_y: %.4f' % train_stat1.mean()
# print 'a_x: %.4f' % train_stat2.mean()
# print 'a_y: %.4f' % train_stat3.mean()
# print 'res_x: %.4f' % train_stat4.mean()
# print 'res_y: %.4f\n' % train_stat5.mean()
#
# output_name.write('alpha_x: %.4f ' % train_stat0.mean())
# output_name.write('alpha_y: %.4f ' % train_stat1.mean())
# output_name.write('a_x: %.4f ' % train_stat2.mean())
# output_name.write('a_y: %.4f ' % train_stat3.mean())
# output_name.write('res_x: %.4f ' % train_stat4.mean())
# output_name.write('res_y: %.4f ' % train_stat5.mean())
# output_name.close()
#
#
# for tag in tags[1:]:
#     output_name = open(output_root + '/' + tag + '_stat_test.txt', 'w')
#     gen_stat = np.load(output_root + '/'  +tag + '_alpha_x.npz')['arr_0']
#     statistic, pvalue = stats.ttest_ind(train_stat0, gen_stat, equal_var=False)
#     print tag
#     print 'alpha_x: %.4f, t-statistics: %.4f, p-value: %.4f' \
#            %(gen_stat.mean(), statistic, pvalue)
#     output_name.write('alpha_x\n')
#     output_name.write('average: %.4f, ' % gen_stat.mean())
#     output_name.write('statistic: %.4f, pvalue: %.4f\n' % (statistic, pvalue))
#
#     gen_stat = np.load(output_root + '/'  +tag + '_alpha_y.npz')['arr_0']
#     statistic, pvalue = stats.ttest_ind(train_stat1, gen_stat, equal_var=False)
#     print 'alpha_y: %.4f, t-statistics: %.4f, p-value: %.4f' \
#            %(gen_stat.mean(), statistic, pvalue)
#     output_name.write('alpha_y\n')
#     output_name.write('average: %.4f, ' % gen_stat.mean())
#     output_name.write('statistic: %.4f, pvalue: %.4f\n' % (statistic, pvalue))
#
#     gen_stat = np.load(output_root + '/'  +tag + '_a_x.npz')['arr_0']
#     statistic, pvalue = stats.ttest_ind(train_stat2, gen_stat, equal_var=False)
#     print 'a_x: %.4f, t-statistics: %.4f, p-value: %.4f' \
#            %(gen_stat.mean(), statistic, pvalue)
#     output_name.write('a_x\n')
#     output_name.write('average: %.4f, ' % gen_stat.mean())
#     output_name.write('statistic: %.4f, pvalue: %.4f\n' % (statistic, pvalue))
#
#     gen_stat = np.load(output_root + '/'  +tag + '_a_y.npz')['arr_0']
#     statistic, pvalue = stats.ttest_ind(train_stat3, gen_stat, equal_var=False)
#     print 'a_y: %.4f, t-statistics: %.4f, p-value: %.4f' \
#            %(gen_stat.mean(), statistic, pvalue)
#     output_name.write('a_y\n')
#     output_name.write('average: %.4f, ' % gen_stat.mean())
#     output_name.write('statistic: %.4f, pvalue: %.4f\n' % (statistic, pvalue))
#
#     gen_stat = np.load(output_root + '/'  +tag + '_res_x.npz')['arr_0']
#     statistic, pvalue = stats.ttest_ind(train_stat4, gen_stat, equal_var=False)
#     print 'res_x: %.4f, t-statistics: %.4f, p-value: %.4f' \
#            %(gen_stat.mean(), statistic, pvalue)
#     output_name.write('res_x\n')
#     output_name.write('average: %.4f, ' % gen_stat.mean())
#     output_name.write('statistic: %.4f, pvalue: %.4f\n' % (statistic, pvalue))
#
#     gen_stat = np.load(output_root + '/'  +tag + '_res_y.npz')['arr_0']
#     statistic, pvalue = stats.ttest_ind(train_stat5, gen_stat, equal_var=False)
#     print 'res_y: %.4f, t-statistics: %.4f, p-value: %.4f' \
#            %(gen_stat.mean(), statistic, pvalue)
#     output_name.write('res_y\n')
#     output_name.write('average: %.4f, ' % gen_stat.mean())
#     output_name.write('statistic: %.4f, pvalue: %.4f\n' % (statistic, pvalue))
#
#     output_name.close()