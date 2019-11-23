# Statistics of Deep Generated Images
Code for the paper ["Statistics of Deep Generated Images"](https://arxiv.org/abs/1708.02688), which is under review as a journal paper.

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
