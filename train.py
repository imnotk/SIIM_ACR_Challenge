import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.backends.cudnn as  cudnn

from dataset import SIIM
from dataset import augmentation
import argparse
import shutil
import numpy as np

# from segmentation_models_pytorch import Unet, Linknet, FPN, PSPNet
# from segmentation_models_pytorch.encoders import get_preprocessing_fn

from segmentation_models.encoders import get_preprocessing_fn
from model import unet

from utils import loss
from utils import lovasz_losses
from utils import metrics
import time
import os

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
parser.add_argument('--use_total',type=bool,default=False,help='pretrained model')
parser.add_argument('--resume',type=bool,default=False,help='whether to load ckpt')
parser.add_argument('--lovasz',type=bool,default=False,help='whether to use lovasz hinge loss')
parser.add_argument('--dr',type=float,default=0.4,help='dropout rate')
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
    split = args.split
    num_workers = args.num_workers
    scheduler_step = args.epochs

    print('using split %d' % split)

    if args.model == 'resnet34unet':
        model = unet.UNet(3,1,activation= None,dr=args.dr)
        preprocess = get_preprocessing_fn('resnet34')
    else:
        raise ValueError(args.model,' is not in our model')

    if os.path.exists(os.path.join('ckpt',args.model,str(args.split))) is False:
        os.makedirs(os.path.join('ckpt',args.model,str(args.split)))

    if os.path.exists(os.path.join('logdir',args.model,str(args.split))) is False:
        os.makedirs(os.path.join('logdir',args.model,str(args.split)))

    train_aug = augmentation.get_augmentations('train',1.0, args.image_size)
    valid_aug = augmentation.get_augmentations('valid',1.0, args.image_size)

    train_dataset = SIIM.SIIM_ACR(
        mode='train',split=split,preprocess=preprocess, augmentation=train_aug,
    )
    
    

    valid_dataset = SIIM.SIIM_ACR(
        mode='valid',split=split,preprocess=preprocess, augmentation=valid_aug,
    )

    print('train valid dataset init successfully')

    if args.use_total:
        train_dataset = SIIM.SIIM_ACR(
        mode='train',split=split,preprocess=preprocess, augmentation=train_aug, use_total=True

    )
    
    # weights = train_dataset.weight + 1
    # train_sampler = WeightedRandomSampler(weights, num_samples=len(train_dataset))

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )

    val_dataloader = DataLoader(
        valid_dataset,
        batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    model = nn.DataParallel(model, device_ids=args.gpus).cuda()

    global metric
    metric = metrics.FscoreMetric(threshold=0.5)
    
    cudnn.benchmark = True

    criterion_1 = loss.BCEDiceLoss().cuda()
    # criterion_1 = nn.BCEWithLogitsLoss()
    # criterion_1 = loss.WeightedBCELoss().cuda()
    # criterion_1 = loss.DiceLoss().cuda()
    criterion_2 = lovasz_losses.lovasz_hinge

    if args.optim == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            args.learning_rate,
            momentum=0.9,
            weight_decay=1e-4)
    elif args.optim == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            args.learning_rate
        )


    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                              T_max= args.epochs,
                                                              eta_min=args.min_lr)
    
    if args.pretrained_model:
            ckpt = torch.load(args.pretrained_model)
            model.load_state_dict(ckpt)
            print('use pretrained model on', args.pretrained_model)
    
    i = 0
    if args.use_total and args.resume:
        if os.path.exists(os.path.join('ckpt', args.model)):
            print('=> loading checkpoint {}'.format(os.path.join('ckpt', args.model)))
            ckpt = torch.load(
                os.path.join('ckpt', args.model, str(i) + '_size%d' % args.image_size + '.pth'))

            model.load_state_dict(ckpt)
        else:
            print('=> no checkpoint found at {}'.format(
                os.path.join('ckpt', args.model, str(i) + '_size%d' % args.image_size + '.pth')))

    elif args.resume:
        if os.path.exists(os.path.join('ckpt', args.model, str(args.split))):
            print('=> loading checkpoint {}'.format(os.path.join('ckpt', args.model, str(args.split))))
            ckpt = torch.load(
                os.path.join('ckpt', args.model, str(args.split), str(i) + '_size%d' % args.image_size + '.pth'))

            model.load_state_dict(ckpt)
        else:
            print('=> no checkpoint found at {}'.format(
                os.path.join('ckpt', args.model, str(i) + '_size%d' % args.image_size + '.pth')))

    for i in range(args.snapshot):

        for epoch in range(args.epochs):

            if args.lovasz is False:
                train(train_dataloader, model, criterion_1, optimizer, epoch, i)
            else:
                train(train_dataloader, model, criterion_2, optimizer, epoch, i)

            lr_scheduler.step()
            if (epoch + 1) % args.val_freq == 0 or epoch == args.epochs - 1:

                if args.lovasz is False:
                    prec1 = validate(val_dataloader, model, criterion_1, epoch, i)
                else:
                    prec1 = validate(val_dataloader, model, criterion_2, epoch, i)

                if prec1 > best_prec1:
                    best_prec1 = prec1
                    best_param = model.state_dict()

                    if args.use_total:
                        torch.save(best_param,
                               os.path.join('ckpt',args.model, str(i) + '_size%d' % args.image_size + '.pth'))
                    else:
                        torch.save(best_param,
                                os.path.join('ckpt',args.model,str(args.split), str(i) + '_size%d' % args.image_size + '.pth'))

        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9,
                                    weight_decay=1e-4)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, scheduler_step, args.min_lr)
        best_prec1 = 0

def train(data_loader, model, criterion, optim, epoch, snap_shot):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    dice_cof = AverageMeter()

    if args.fine_tune:
        model.eval()
    else:
        model.train()
    
    end = time.time()
    for i, (inputs, label) in enumerate(data_loader):


        data_time.update(time.time() - end)

        label = torch.autograd.Variable(label['mask'])
        input_var = inputs.cuda(async=True)
        target_var = label.cuda(async=True)
        output = model.forward(input_var)
        loss = criterion(output, target_var)

        score = dice_overall_v2(output, target_var)
        
        loss = loss / args.accumulate_step
        loss.backward()

        if (i+1) % args.accumulate_step == 0:
            optim.step()
            optim.zero_grad()

        losses.update(loss.data, inputs.size(0))
        dice_cof.update(score.data, inputs.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print(('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Dice_coff {dice_cof.val:.3f} ({dice_cof.avg:.3f})\t'.format(
                   epoch, i, len(data_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, dice_cof=dice_cof, lr=optim.param_groups[-1]['lr'])))

    if (epoch + 1) % args.val_freq == 0 or epoch == args.epochs - 1:

        template = 'Epoch: {}, Loss {:.2f}, Dice_cof@1 {:.3f}\n'
        if args.use_total:
            with open(os.path.join('logdir', args.model, 'train_txt_%d_%d.txt' % (snap_shot, args.image_size)), 'a') as f:
                f.writelines(
                    template.format(
                        epoch, losses.avg, dice_cof.avg
                    )
                )
        else:
            with open(os.path.join('logdir', args.model, str(args.split),'train_txt_%d_%d.txt' % (snap_shot, args.image_size)), 'a') as f:
                f.writelines(
                    template.format(
                        epoch, losses.avg, dice_cof.avg
                    )
                )

def validate(data_loader, model, criterion, epoch, snap_shot):

    model.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    dice_cof = AverageMeter()

    with torch.no_grad():
        end = time.time()
        for i, (inputs, label) in enumerate(data_loader):

            label = label['mask']
            input_var = inputs.cuda(async=True)
            target_var = label.cuda(async=True)

            output = model(input_var)
            loss = criterion(output, target_var)

            score = dice_overall_v2(output.data, target_var)

            batch_time.update(time.time() - end)
            end = time.time()

            losses.update(loss.data, inputs.size(0))
            dice_cof.update(score.data, inputs.size(0))

            if i % args.print_freq == 0:
                print(('Test: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Dice_coff {dice_cof.val:.3f} ({dice_cof.avg:.3f})\t'.format(
                    i, len(data_loader), batch_time=batch_time, loss=losses,
                    dice_cof=dice_cof)))

        print(('Testing Results: Dice_coff@1 {dice_cof.avg:.3f} Loss {loss.avg:.5f}'
            .format(dice_cof=dice_cof, loss=losses)))

        template = 'Epoch: {}, Loss {:.2f}, Dice_cof@1 {:.3f}\n'
        if args.use_total:
            with open(os.path.join('logdir', args.model, 'valid_txt_%d_%d.txt' % (snap_shot, args.image_size)), 'a') as f:
                f.writelines(
                    template.format(
                        epoch, losses.avg, dice_cof.avg
                    )
                )
        else:
            with open(os.path.join('logdir', args.model, str(args.split),'valid_txt_%d_%d.txt' % (snap_shot, args.image_size)), 'a') as f:
                f.writelines(
                    template.format(
                        epoch, losses.avg, dice_cof.avg
                    )
                )

    return dice_cof.avg

def dice_overall_v2(preds, targs):
    preds = torch.sigmoid(preds)
    preds = (preds > 0.5).float()
    n = preds.shape[0]
    preds = preds.view(n, -1)
    targs = targs.view(n, -1)
    intersect = (preds * targs).sum(-1).float()
    union = (preds+targs).sum(-1).float()
    u0 = union==0
    intersect[u0] = 1
    union[u0] = 2
    return (2. * intersect / union).mean()

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