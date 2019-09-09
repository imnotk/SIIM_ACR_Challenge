import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.backends.cudnn as  cudnn

from dataset import SIIM
from dataset import augmentation

import argparse
import tqdm
import cv2
import numpy as np
from utils import loss
from utils import lovasz_losses
from utils import metrics
import time
import os
from segmentation_models.encoders import get_preprocessing_fn
from model import unet

parser = argparse.ArgumentParser('mynetwork')

parser.add_argument('-b', '--batch_size', type=int, default=16, help='batch_size we used')
parser.add_argument('-val_freq', '--val_freq', type=int, default=5, help='validataion every k epochs')
parser.add_argument('-print_freq', '--print_freq', type=int, default=5, help='print logging file every k epochs')
parser.add_argument('-num_workers', type=int, default=8, help='dataset loading workers')
parser.add_argument('--split', type=int, default=1, choices=[0, 1, 2, 3, 4], help='which split to train')
parser.add_argument('--model', type=str, default='resnet34unet', help='training model')
parser.add_argument('--resume', type=bool, default=False, help='whether to load ckpt')
parser.add_argument('--use_flip', type=bool, default=False, help='whether to load ckpt')
parser.add_argument('--image_size', type=int, default=224, help='image_size')
parser.add_argument('--gpus', type=int, nargs='+', default=None, help='gpus')
parser.add_argument('--snapshot', default=2, type=int, help='Number of snapshots per fold')


args = parser.parse_args()

best_prec1 = 0


def main():
    global best_prec1

    batch_size = args.batch_size
    split = args.split
    num_workers = args.num_workers


    if args.model == 'resnet34unet':
        model = unet.UNet(3,1,activation= 'sigmoid',weight=None)
        preprocess = get_preprocessing_fn('resnet34')
    else:
        raise ValueError(args.model,' is not in our model')

    

    valid_aug = augmentation.get_augmentations('valid', 1.0, args.image_size)


    valid_dataset = SIIM.SIIM_ACR(
        mode='valid', split=split, preprocess=preprocess, augmentation=valid_aug
    )


    val_dataloader = DataLoader(
        valid_dataset,
        batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    if args.use_flip:
        valid_dataset_tta = SIIM.SIIM_ACR(
            mode='valid', split=split, preprocess=preprocess, augmentation=valid_aug, is_tta=True
        )


        global val_dataloader_tta
        val_dataloader_tta = DataLoader(
            valid_dataset_tta,
            batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
        )

    model = nn.DataParallel(model, device_ids=args.gpus).cuda()

    global metric
    metric = metrics.FscoreMetric(threshold=None)

    for i in range(args.snapshot):

        if os.path.exists(os.path.join('ckpt', args.model, str(args.split))):
            print('=> loading checkpoint {}'.format(os.path.join('ckpt', args.model, str(args.split))))
            ckpt = torch.load(os.path.join('ckpt', args.model, str(args.split),str(i) + '_size%d' % args.image_size + '.pth'))

            model.load_state_dict(ckpt)
        else:
            raise ValueError('=> no checkpoint found at {}'.format(os.path.join('ckpt', args.model)))



        prec1 = validate(val_dataloader, model, i)



def validate(data_loader, model, snapshot):
    model.eval()

    batch_time = AverageMeter()
    preds = []
    targ = []

    with torch.no_grad():
        end = time.time()
        for i, (inputs, label) in enumerate(data_loader):
            
            

            label = label.cuda(async=True)
            input_var = inputs.cuda()

            output = model(input_var)
            
            # out = norm_pred(output.cpu().numpy())

            preds.append(output.cpu().numpy())
            targ.append(label.cpu().numpy())

            del output, input_var
            batch_time.update(time.time() - end)
            end = time.time()
            print('step %d use batch_time %.2f' % (i,batch_time.avg))

        preds = np.concatenate(preds,axis=0)
        ys = np.concatenate(targ,axis=0)

    # preds = norm_pred(preds)

    print('validate procession over')
    
    if args.use_flip:
        with torch.no_grad():
            end = time.time()
            preds_tta = []
            for i, (inputs, label) in enumerate(val_dataloader_tta):

                label = label.cuda(async=True)
                input_var = inputs.cuda()

                output = model(input_var)

                preds_tta.append(output.cpu().numpy()[...,::-1])

                del output, input_var, label
                batch_time.update(time.time() - end)
                end = time.time()
                print('step %d use batch_time %.2f' % (i,batch_time.avg))

            preds_tta = np.concatenate(preds_tta,axis=0)
            preds = (preds + preds_tta) // 2
            # preds = preds_tta

    dices = []
    thrs = np.arange(0, 1, 0.1)
    for i in tqdm.tqdm(thrs):
        preds_m = np.float32(preds > i)
        dices.append(np.mean(dice_overall(preds_m, ys)))
    dices = np.array(dices)
    print(dices)

    template = 'At threshold {:.2f}, Dice_cof@1 {:.4f}\n'
    with open(
            os.path.join('logdir', args.model, str(args.split), 'valid_result_snap%d_size%d.txt' % (snapshot,args.image_size)), 'a') as f:
        f.writelines(
            template.format(
                    np.argmax(dices) * 0.1, np.max(dices)
            )
        )


def dice_overall(preds, targs):
    n = preds.shape[0]
    preds = np.reshape(preds,(n,-1))
    targs = np.reshape(targs,(n,-1))
    intersect = np.sum(preds * targs, axis=-1)
    union = np.sum(preds + targs, axis=-1)
    u0 = union==0
    intersect[u0] = 1
    union[u0] = 2
    return (2. * intersect / union)

def dice_overall_v2(preds, targs):
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


# def accuracy(output, target, topk=(1,)):
#     maxk = max(topk)
#     batch_size = target.size(0)
#     _, pred = output.topk(maxk, 1, True, True)
#     pred = pred.t()
#     correct = pred.eq(target.view(1,-1).expand_as(pred))

#     res = []
#     for k in topk:
#         correct_k = correct[:k].view(-1).float().sum(0)
#         res.append(correct_k.mul_(100 / batch_size))

#     return res

if __name__ == "__main__":
    main()