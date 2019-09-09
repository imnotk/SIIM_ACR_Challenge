import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.backends.cudnn as  cudnn

from dataset import SIIM, SIIM_test
from dataset import augmentation

import argparse
import tqdm
from tqdm import tqdm_notebook, tqdm
import pandas as pd
import cv2

import numpy as np
from utils import loss
from utils import lovasz_losses
from utils import metrics
import time
import os
from segmentation_models.encoders import get_preprocessing_fn
from model import unet

from data import mask_functions

parser = argparse.ArgumentParser('mynetwork')

parser.add_argument('-b','--batch_size',type=int,default=32,help='batch_size we used')
parser.add_argument('-lr','--learning_rate',type=float,default=1e-2,help='learning_rate for the optim')
parser.add_argument('--min_lr',type=float,default=1e-3,help='learning_rate for the optim')
parser.add_argument('--epochs',type=int,default=50,help='training epoch')
parser.add_argument('--val_freq',type=int,default=5,help='validataion every k epochs')
parser.add_argument('--print_freq',type=int,default=5,help='print logging file every k epochs')
parser.add_argument('--num_workers',type=int,default=8,help='dataset loading workers')
parser.add_argument('--split',type=int,default=1,choices=[0,1,2,3,4],help='which split to train')
parser.add_argument('--model',type=str,default='resnet34unet',help='training model')
parser.add_argument('--pretrained_model',type=str,help='pretrained model')
parser.add_argument('--use_total',type=str,default=False,help='pretrained model')
parser.add_argument('--resume',type=bool,default=False,help='whether to load ckpt')
parser.add_argument('--lovasz',type=bool,default=False,help='whether to use lovasz hinge loss')
parser.add_argument('--dr',type=float,default=0.5,help='dropout rate')
parser.add_argument('--threshold',type=float,default=0.5,help='dropout rate')
parser.add_argument('--image_size',type=int,default=256,help='image_size')
parser.add_argument('--gpus',type=int,nargs='+',default=None,help='gpus')
parser.add_argument('--optim',type=str,default='sgd',choices=['adam','sgd'],help='gpus')
parser.add_argument('--snapshot', default=2, type=int, help='Number of snapshots per fold')
parser.add_argument('--accumulate_step', default=1, type=int, help='accumulate gradients step')
parser.add_argument('--fine_tune',type=bool,default=False,help='whether finetune use eval mode')


args = parser.parse_args()

best_prec1 = 0


def main():
    global best_prec1

    batch_size = args.batch_size
    lr = args.learning_rate
    epochs = args.epochs
    val_freq = args.val_freq
    split = args.split
    num_workers = args.num_workers

    if os.path.exists(os.path.join('logdir', args.model, str(args.split))) is False:
        os.makedirs(os.path.join('logdir', args.model, str(args.split)))

    if args.model == 'resnet34unet':
        model = unet.UNet(3,1,activation= 'sigmoid',weight=None)
        preprocess = get_preprocessing_fn('resnet34')
    else:
        raise ValueError(args.model,' is not in our model')

    valid_aug = augmentation.get_augmentations('valid', 1.0, args.image_size)


    valid_dataset = SIIM_test.SIIM_ACR(
        mode='test', preprocess=preprocess, augmentation=valid_aug
    )

    val_dataloader = DataLoader(
        valid_dataset,
        batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    model = nn.DataParallel(model, device_ids=args.gpus).cuda()

    global metric
    metric = metrics.FscoreMetric(threshold=None)

    # cudnn.benchmark = True

    for i in range(args.snapshot):

        if args.pretrained_model:
            ckpt = torch.load(args.pretrained_model)
            model.load_state_dict(ckpt)
            print('use pretrained model on', args.pretrained_model)
        elif args.use_total:
            print('=> loading checkpoint {}'.format(os.path.join('ckpt', args.model)))
            ckpt = torch.load(
                os.path.join('ckpt', args.model, str(i) + '_size%d' % args.image_size + '.pth'))
            
            model.load_state_dict(ckpt)
        elif os.path.exists(os.path.join('ckpt', args.model, str(args.split))):
            print('=> loading checkpoint {}'.format(os.path.join('ckpt', args.model, str(args.split))))
            ckpt = torch.load(
                os.path.join('ckpt', args.model, str(args.split), str(i) + '_size%d' % args.image_size + '.pth'))

            model.load_state_dict(ckpt)
        
        else:
            print('=> no checkpoint found at {}'.format(os.path.join('ckpt', args.model)))

        

        prec1 = test(val_dataloader, model, i)






def test(data_loader, model, snapshot):
    model.eval()

    batch_time = AverageMeter()

    rles = []
    ids = []
    masks = []
    threshold = args.threshold
    with torch.no_grad():
        end = time.time()
        for i, (inputs, imageid) in enumerate(data_loader):

            input_var = inputs.cuda()

            output = model(input_var)

            output = output.squeeze()
            masks.append(output.cpu().numpy())
            ids.extend(imageid)

            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print((
                       'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'.format(
                     batch_time=batch_time)))

    # masks = torch.stack(masks,dim=0)
    # masks = masks.cpu().numpy()
    masks = np.stack(masks,axis=0)
    print(masks.shape)
    for p in tqdm(masks):
        im = cv2.resize(p, (1024, 1024))
        im = (im[:, :] > threshold).astype(np.uint8).T
        rles.append(mask_functions.mask2rle(im, 1024, 1024))

    sub_df = pd.DataFrame({'ImageId':ids, 'EncodedPixels':rles})
    sub_df.loc[sub_df.EncodedPixels=='','EncodedPixels'] = '-1'
    if args.use_total:
        sub_df.to_csv(os.path.join('logdir',args.model,'submission_snap%d_size%d.csv' % (snapshot,args.image_size)),index=False)
    else:
        sub_df.to_csv(os.path.join('logdir',args.model,str(args.split),'submission_snap%d_size%d.csv' % (snapshot,args.image_size)),index=False)

    print(sub_df.tail(10))


def dice_overall(preds, targs):
    n = preds.shape[0]
    preds = preds.view(n, -1)
    targs = targs.view(n, -1)
    intersect = (preds * targs).sum(-1).float()
    union = (preds+targs).sum(-1).float()
    u0 = union==0
    intersect[u0] = 1
    union[u0] = 2
    return (2. * intersect / union)

def adjust_learning_rate(optim, epoch, lr_step):
    lr = args.learning_rate * 0.1 ** (sum(epoch >= np.array(lr_step)))

    for param_group in optim.param_groups:
        param_group['lr'] = lr


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == "__main__":
    main()