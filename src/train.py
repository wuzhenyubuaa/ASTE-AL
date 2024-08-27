#!/usr/bin/python3
#coding=utf-8
import sys
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import dataset
from net  import F3Net
from apex import amp
import os
import numpy as np
from argparse import ArgumentParser
from utils import Visualiser, query_strategy
from query import QuerySelector
import gc
import cv2
from utils_tools.CCLS import CyclicCosineAnnealingLR
from art.attacks.evasion import ProjectedGradientDescent

sys.path.insert(0, '../')
sys.dont_write_bytecode = True
envpath = '/home/jack/anaconda3/lib/python3.8/site-packages/cv2/qt/plugins/platforms'
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = envpath
parser = ArgumentParser('active learning')


"""Parameters configuration of active learning algorithm."""
parser.add_argument('--n_pixels_per_img', type=int, default=10)
parser.add_argument('--use_superpixel', type=bool, default=True)
parser.add_argument('--point_supervise', type=bool, default=True) # point supervise or full-supervise
parser.add_argument('--max_budget', type=int, default=100)
parser.add_argument('--ignore_index', type=int, default=255)
parser.add_argument('--query_strategy', type=str, default='mvmc',
                    choices=["least_confidence", "mvmc", "entropy", "random"])
parser.add_argument('--query', type=bool, default=True)
parser.add_argument('--top_n_percent', type=float, default=0.05)
# parser.add_argument('--box_point', type=bool, default=False) # using box supervision or not


"""Parameters configuration of training setting"""
parser.add_argument('--dir_dataset', type=str, default='/media/jack/新加卷/data/sod/dut_tr')
parser.add_argument('--dir_checkpoints', type=str, default='../experiments/model-1')
parser.add_argument('--batchsize', type=int, default=8)
parser.add_argument('--epoch', type=int, default=32)
parser.add_argument('--lr', type=float, default=0.05)
parser.add_argument('--moment', type=float, default=0.9)
parser.add_argument('--decay', type=float, default=0.0005)
parser.add_argument('--num_works', type=int, default=8)
parser.add_argument('--output_channel', type=int, default=2)  # full-supervised: 1, point-supervised: 2
parser.add_argument('--mode', type=str, default='train')

args = parser.parse_args()


class Logger():
    def __init__(self, filepath='./'):
        self.terminal = sys.stdout
        self.log = open(filepath+"log.txt", "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

os.makedirs(args.dir_checkpoints, exist_ok=True)
sys.stdout = Logger(filepath=args.dir_checkpoints)


def structure_loss(pred, mask):
    weit  = 1+5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15)-mask)
    wbce  = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce  = (weit*wbce).sum(dim=(2,3))/weit.sum(dim=(2,3))
    pred  = torch.sigmoid(pred)
    inter = ((pred*mask)*weit).sum(dim=(2,3))
    union = ((pred+mask)*weit).sum(dim=(2,3))
    wiou  = 1-(inter+1)/(union-inter+1)
    return (wbce+wiou).mean()


''' Full-supervised model training pipeline '''
def train_full(Dataset, Network):
    ## dataset
    cfg    = Dataset.Config(datapath=args.dir_dataset, savepath=args.dir_checkpoints,
                            mode=args.mode, batch=args.batchsize,
                            lr=args.lr, momen=args.moment, decay=args.decay,
                            epoch=args.epoch, output_channel= args.output_channel)
    data   = Dataset.Data(cfg, args)
    loader = DataLoader(data, collate_fn=data.collate, batch_size=cfg.batch, shuffle=True, num_workers=args.num_works)
    ## network
    net    = Network(cfg)
    net.train(True)
    net.cuda()
    ## parameter
    base, head = [], []
    for name, param in net.named_parameters():
        if 'bkbone.conv1' in name or 'bkbone.bn1' in name:
            print(name)
        elif 'bkbone' in name:
            base.append(param)
        else:
            head.append(param)
    optimizer      = torch.optim.SGD([{'params':base}, {'params':head}], lr=cfg.lr, momentum=cfg.momen, weight_decay=cfg.decay, nesterov=True)
    net, optimizer = amp.initialize(net, optimizer, opt_level='O2')
    sw             = SummaryWriter(cfg.savepath)
    global_step    = 0

    for epoch in range(cfg.epoch):
        optimizer.param_groups[0]['lr'] = (1-abs((epoch+1)/(cfg.epoch+1)*2-1))*cfg.lr*0.1
        optimizer.param_groups[1]['lr'] = (1-abs((epoch+1)/(cfg.epoch+1)*2-1))*cfg.lr

        for step, (image, mask) in enumerate(loader):

            image, mask = image.cuda().float(), mask.cuda().float()
            out1u, out2u, out2r, out3r, out4r, out5r = net(image)

            loss1u = structure_loss(out1u, mask)
            loss2u = structure_loss(out2u, mask)
            loss2r = structure_loss(out2r, mask)
            loss3r = structure_loss(out3r, mask)
            loss4r = structure_loss(out4r, mask)
            loss5r = structure_loss(out5r, mask)
            loss   = (loss1u+loss2u)/2+loss2r/2+loss3r/4+loss4r/8+loss5r/16

            optimizer.zero_grad()
            with amp.scale_loss(loss, optimizer) as scale_loss:
                scale_loss.backward()
            optimizer.step()

            ## log
            global_step += 1
            sw.add_scalar('lr'   , optimizer.param_groups[0]['lr'], global_step=global_step)
            sw.add_scalars('loss', {'loss1u':loss1u.item(), 'loss2u':loss2u.item(),
                                    'loss2r':loss2r.item(), 'loss3r':loss3r.item(),
                                    'loss4r':loss4r.item(), 'loss5r':loss5r.item()}, global_step=global_step)
            if step%10 == 0:
                print('%s | step:%d/%d/%d | lr=%.6f | loss=%.6f'%(datetime.datetime.now(),
                                                                  global_step, epoch+1, cfg.epoch,
                                                                  optimizer.param_groups[0]['lr'], loss.item()))

        if epoch==cfg.epoch:
            torch.save(net.state_dict(), cfg.savepath+'/model-'+str(epoch+1))

def point_loss(pred, mask):
    ce = F.cross_entropy(pred, mask.squeeze(1).long(), ignore_index=args.ignore_index)
    return ce




''' Point-supervised model training pipeline '''
def train_point(Dataset, Network):

    ## network
    net = Network(cfg)
    net.cuda()

    ## dataset
    cfg    = Dataset.Config(datapath=args.dir_dataset, savepath=args.dir_checkpoints,
                            mode='train', batch=args.batchsize,
                            lr=args.lr, momen=args.moment, decay=args.decay,
                            epoch=args.epoch, output_channel= args.output_channel)
    data   = Dataset.Data(cfg, args)

    """Stage 1: FPGD attack"""
    pgd = ProjectedGradientDescent(estimator=net, eps=0.1, eps_step=0.01, max_iter=5)
    data = pgd.generate(x=data)
    dataloader = DataLoader(data, collate_fn=data.collate, batch_size=cfg.batch, shuffle=True, num_workers=args.num_works, drop_last=True)
    data = Dataset.Data(cfg, args, query=True)
    dataloader_query = DataLoader(data, batch_size=1, shuffle=False, num_workers=args.num_works)

    ## parameter
    base, head = [], []
    for name, param in net.named_parameters():
        if 'bkbone.conv1' in name or 'bkbone.bn1' in name:
            print(name)
        elif 'bkbone' in name:
            base.append(param)
        else:
            head.append(param)
    optimizer      = torch.optim.SGD([{'params':base}, {'params':head}], lr=cfg.lr, momentum=cfg.momen, weight_decay=cfg.decay, nesterov=True)
    net, optimizer = amp.initialize(net, optimizer, opt_level='O2')
    sw             = SummaryWriter(cfg.savepath)
    global_step    = 0


    """ Stage 2: (b) """
    n_stages = args.max_budget // args.n_pixels_per_img
    print('n_stages:', n_stages)

    for nth_query in range(n_stages):

        net.train(True)
        os.makedirs(f'{args.dir_checkpoints}/query_{nth_query}', exist_ok=True)

        for epoch in range(cfg.epoch):
            optimizer.param_groups[0]['lr'] = (1-abs((epoch+1)/(cfg.epoch+1)*2-1))*cfg.lr*0.1
            optimizer.param_groups[1]['lr'] = (1-abs((epoch+1)/(cfg.epoch+1)*2-1))*cfg.lr
            gc.collect()

            for step, (image, mask, queries, name) in enumerate(dataloader):

                image, mask, queries = image.cuda().float(), mask.cuda().float(), queries.cuda()
                img, gt = image, mask
                mask.flatten()[~queries.flatten()] = args.ignore_index
                out1u, out2u, out2r, out3r, out4r, out5r = net(image)

                loss1u = point_loss(out1u, mask)
                loss2u = point_loss(out2u, mask)
                loss2r = point_loss(out2r, mask)
                loss3r = point_loss(out3r, mask)
                loss4r = point_loss(out4r, mask)
                loss5r = point_loss(out5r, mask)
                loss   = (loss1u+loss2u)/2+loss2r/2+loss3r/4+loss4r/8+loss5r/16

                optimizer.zero_grad()
                with amp.scale_loss(loss, optimizer) as scale_loss:
                    scale_loss.backward()
                optimizer.step()

                ## log
                global_step += 1
                sw.add_scalar('lr'   , optimizer.param_groups[0]['lr'], global_step=global_step)
                sw.add_scalars('loss', {'loss1u':loss1u.item(), 'loss2u':loss2u.item(),
                                        'loss2r':loss2r.item(), 'loss3r':loss3r.item(),
                                        'loss4r':loss4r.item(), 'loss5r':loss5r.item()}, global_step=global_step)
                if step%10 == 0:
                    print('%s | nth_query:%d | step:%d/%d/%d | lr=%.6f | loss=%.6f'%(datetime.datetime.now(),
                                                                                     nth_query ,global_step, epoch+1,
                                                                                     cfg.epoch, optimizer.param_groups[0]['lr'], loss.item()))
                gc.collect()

            # save the intermedia training data result
            print(name[0])
            prob,  pred = F.softmax(out2u.detach(), dim=1), out2u.argmax(dim=1)
            ent, lc, ms = [query_strategy(prob, uc)[0].cpu() for uc in ["entropy", "least_confidence", "mvmc"]]
            dict_tensors = {'input': img[0].cpu(),
                                'target': gt[0].cpu(),
                                'pred': pred[0].detach().cpu(),
                                'confidence': lc,
                                'margin': -ms,  # minus sign is to draw smaller margin part brighter
                                'entropy': ent}
            file_name = f'{args.dir_checkpoints}/query_{nth_query}/train_{epoch}.png'
            Visualiser()(dict_tensors=dict_tensors, fp=file_name)


            # Save some intermediate results for display purposes
            save = False
            if save:
                from dataset import Normalize, Resize, ToTensor
                os.makedirs(f'{args.dir_checkpoints}/query_{nth_query}/epoch_{epoch}', exist_ok=True)
                mean = np.array([[[124.55, 118.90, 102.94]]])
                std = np.array([[[56.77, 55.97, 57.50]]])
                image_name_list=['ILSVRC2012_test_00000172', 'ILSVRC2012_test_00001213', 'ILSVRC2012_test_00003238', 'ILSVRC2012_test_00004254',
                                 'ILSVRC2012_test_00004377', 'ILSVRC2012_test_00007730', 'ILSVRC2012_test_00008808', 'ILSVRC2012_test_00009334',
                                 'ILSVRC2012_test_00010064', 'ILSVRC2012_test_00011802', 'ILSVRC2012_test_00025895', 'ILSVRC2012_test_00026522',
                                 'ILSVRC2012_test_00031365', 'ILSVRC2012_test_00033063', 'ILSVRC2012_test_00035010',  'ILSVRC2012_test_00035586',
                                 'ILSVRC2012_test_00040141', 'ILSVRC2012_test_00041200', 'ILSVRC2012_test_00042756', 'ILSVRC2012_test_00043508',
                                 'ILSVRC2012_test_00043567', 'ILSVRC2012_test_00044945', 'ILSVRC2012_test_00045939', 'ILSVRC2012_test_00049473',
                                 'ILSVRC2012_test_00049610', 'ILSVRC2012_test_00058509', 'ILSVRC2012_test_00063666', 'ILSVRC2012_test_00064202',
                                 'ILSVRC2012_test_00068857', 'ILSVRC2012_test_00089089', 'ILSVRC2013_test_00002744', 'ILSVRC2013_test_00003177']

                for name in image_name_list:

                    image_ = cv2.imread(args.dir_dataset + '/img/' + name + '.jpg')[:, :, ::-1].astype(np.float32)
                    mask_ = cv2.imread(args.dir_dataset + '/gt/' + name + '.png', 0).astype(np.float32)
                    normalize_ = Normalize(mean=mean, std=std)
                    resize_ = Resize(352, 352)
                    totensor_ = ToTensor()

                    image_, mask_ = normalize_(image_, mask_)
                    image_, mask_ = resize_(image_, mask_)
                    image_, mask_ = totensor_(image_, mask_)
                    image_, mask_ = image_.unsqueeze(0).cuda().float(), mask_.unsqueeze(0).cuda().float()
                    _, output_, _, _, _, _ = net(image_)

                    prob, pred = F.softmax(output_.detach(), dim=1), output_.argmax(dim=1)
                    ent, lc, ms = [query_strategy(prob, uc)[0].cpu() for uc in ["entropy", "least_confidence", "mvmc"]]
                    dict_tensors = {'input': image_[0].cpu(),
                                    'target': mask_[0].cpu(),
                                    'pred': pred[0].detach().cpu(),
                                    'confidence': lc,
                                    'margin': -ms,  # minus sign is to draw smaller margin part brighter
                                    'entropy': ent}
                    file_name = f'{args.dir_checkpoints}/query_{nth_query}/epoch_{epoch}/{name}.png'
                    Visualiser()(dict_tensors=dict_tensors, fp=file_name)

            # save the trained model
            if (epoch+1) == args.epoch:
                torch.save(net.state_dict(), f'{args.dir_checkpoints}/query_{nth_query}' + '/model-'+str(epoch+1))

        # update the labeled pixels, select queries using the current model and label them.
        query_selector = QuerySelector(args, dataloader_query)
        queries_list = query_selector(nth_query+1, net)


if __name__=='__main__':
    if args.point_supervise == True:
        args.output_channel=2
        train_point(dataset, F3Net)
    else:
        args.output_channel=1
        train_full(dataset, F3Net)