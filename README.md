# Statistics of Deep Generated Images
Code for the paper "Statistics of Deep Generated Images", which is under review as a journal paper.

## Usage
All models are implemented in Python 2.7 with [Pytorch](https://github.com/pytorch/pytorch). 

run ```train_xxx.py``` to train the models (e.g., ```train_vae.py```)

run ```stats_xxx.py``` to compute the statistics (e.g., run ```stats_rdf.py``` to compute the random filter response distribution.)

run ```print_xxx.py``` to show the results.

## Data
The models can be trained with natural images or cartoons. 

We collect 511,460 cartoon images extracted from 303 video files of 73 cartoon movies.

To enhance diversity of the images, only one frame is extracted per second. 

You can [email me](mailto:zengyu@mail.dlut.edu.cn) to ask for the cartoon images if interested.

# 图片生成模型和训练数据

## 模型
其中VAE和WGAN也用了DCGAN的网络结构

DCGAN和VAE能产生256的图片

WGAN只能128，大了训练不出来

DRAW用的全连接层和LSTM，只能产生64的图片。图大了会报GPU同步错误。原因未知

试了作者版的PixelCNN，只有一个显卡训练起来十分慢，暂时放弃PixelCNN。下次需要它时再试试。

参考：

[pytorch example DCGAN](https://github.com/pytorch/examples/tree/master/dcgan)

[Praveen关于DRAW的博文](https://pravn.wordpress.com/2017/09/11/rnncell-modules-in-pytorch-to-implement-draw/)和他的[DRAW生成手写数字](https://github.com/pravn/vae_draw)


## 数据

照片：ImageNet里找了510530张

卡通：为了防止重复的图片太多，动画片每秒钟只截一张图。总共从73个动画片中得到511460帧。毕竟下载这么多不容易，如果有人有兴趣可以找我要这些图片（虽然应该没有人有兴趣）

猫和老鼠 12589

马男波杰克 19000+16684

斗罗大陆 3809

犬夜叉 9509

死亡笔记 49451

喜羊羊 5645

特别车队 10158

十万个冷笑话 4958

地狱少女 34685

哆啦A梦 4338

双王 588

狐妖小红娘 18702

鹭与鹤 585

哪吒传奇 19850

镇魂街 20576

大闹天宫 6840

妖怪名单 12541

蝙蝠侠：第一年、红帽子、黑暗骑士归来 17535

腌黄瓜先生 2040

伊藤润二精选集 5832

妄想代理人 17752

李献计历险记 1206

油管上面随便找的不认识的（共48个视频） 147030

迪士尼系列 36524

兔八哥 16182

憨豆先生 16854
