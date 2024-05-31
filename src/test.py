#!/usr/bin/python3
#coding=utf-8

import os
import sys
sys.path.insert(0, '../')
sys.dont_write_bytecode = True

import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import dataset
from net  import F3Net

from argparse import ArgumentParser

envpath = '/home/jack/anaconda3/lib/python3.8/site-packages/cv2/qt/plugins/platforms'
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = envpath


parser = ArgumentParser('active learning')

# active learning
parser.add_argument('--n_pixels_per_img', type=int, default=10)
parser.add_argument('--use_superpixel', type=bool, default=True)
parser.add_argument('--point_supervise', type=bool, default=False) # point supervise or full-supervise
parser.add_argument('--max_budget', type=int, default=100)
parser.add_argument('--ignore_index', type=int, default=255)
parser.add_argument('--query_strategy', type=str, default='mvmc',
                    choices=["least_confidence", "mvmc", "entropy", "random"])
parser.add_argument('--query', type=bool, default=True)
parser.add_argument('--top_n_percent', type=float, default=0.05)




# training setting
parser.add_argument('--dir_dataset', type=str, default='/media/jack/新加卷/data/sod/dut_tr')
parser.add_argument('--dir_checkpoints', type=str, default='../experiments/model-1')
parser.add_argument('--batchsize', type=int, default=8)
parser.add_argument('--epoch', type=int, default=1)
parser.add_argument('--lr', type=float, default=0.05)
parser.add_argument('--moment', type=float, default=0.9)
parser.add_argument('--decay', type=float, default=0.0005)
parser.add_argument('--num_works', type=int, default=8)


# testing
parser.add_argument('--mode', type=str, default='test')

'''
根据需求修改下面的参数 
'''
parser.add_argument('--test_dataset_root', type=str, default='/media/jack/ZhenyuWu/mywork/work6/SODGAN/saved_model/biggan/all/model-1/') # the root of testing datasets
parser.add_argument('--model_path', type=str, default='/home/jack/mywork/work7/PointSOD/pretrained_model/model-32') # the saved model
parser.add_argument('--pre_saved_path', type=str, default='/media/jack/ZhenyuWu/mywork/work6/SODGAN/saved_model/biggan/all/model-1/samples_10k/') # the path of predicted saliency maps
parser.add_argument('--output_channel', type=int, default=1)  # full-supervised: 1, point-supervised: 2
args = parser.parse_args()

class Test(object):
    def __init__(self, Dataset, Network, path):
        ## dataset
        self.cfg    = Dataset.Config(datapath=path, snapshot=args.model_path, mode=args.mode, output_channel= args.output_channel)
        self.data   = Dataset.Data(self.cfg, args)
        self.loader = DataLoader(self.data, batch_size=1, shuffle=False, num_workers=8)
        ## network
        self.net    = Network(self.cfg)
        self.net.train(False)
        self.net.cuda()


    
    def __call__(self, ):
        with torch.no_grad():
            for image, mask, shape, name in self.loader:
                image = image.cuda().float()
                out1u, out2u, out2r, out3r, out4r, out5r = self.net(image, shape)

                if args.output_channel == 2: # for point supervision
                    pred = out2u.argmax(dim=1)[0].cpu().numpy()

                    pred = (pred - pred.min()) / (pred.max() - pred.min() ) * 255

                    path = args.pre_saved_path + self.cfg.datapath.split('/')[-1]
                    if not os.path.exists(path):
                        os.makedirs(path)
                    cv2.imwrite(path + '/' + name[0] + '.png', np.round(pred))
                    print(name[0])
                else:#****************************** for full-supervsion
                    out   = out2u
                    pred  = (torch.sigmoid(out[0,0])*255).cpu().numpy()
                    path  = args.pre_saved_path + self.cfg.datapath.split('/')[-1]
                    if not os.path.exists(path):
                        os.makedirs(path)
                    cv2.imwrite(path+'/'+name[0]+'.png', np.round(pred))
                    print(name[0])


if __name__=='__main__':
    datasets = ['samples_10k']
    # datasets = ['ecssd', 'pascal', 'dut_te', 'hku', 'dut_omron']
    for data in datasets:
        Test(dataset, F3Net, args.test_dataset_root + data)()

