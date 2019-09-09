import torch
import torch.nn as nn
from . import functions as F

class JaccardLoss(nn.Module):
    __name__ = 'jaccard_loss'

    def __init__(self, eps=1e-7, activation='sigmoid'):
        super().__init__()
        self.activation = activation
        self.eps = eps

    def forward(self, y_pr, y_gt):
        return 1 - F.jaccard(y_pr, y_gt, eps=self.eps, threshold=None, activation=self.activation)


class DiceLoss(nn.Module):
    __name__ = 'dice_loss'

    def __init__(self, eps=1e-7, activation='sigmoid'):
        super().__init__()
        self.activation = activation
        self.eps = eps

    def forward(self, y_pr, y_gt):
        return 1 - F.f_score(y_pr, y_gt, beta=1., eps=self.eps, threshold=None, activation=self.activation)


class BCEJaccardLoss(JaccardLoss):
    __name__ = 'bce_jaccard_loss'

    def __init__(self, eps=1e-7, activation='sigmoid'):
        super().__init__(eps, activation)
        self.bce = nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self, y_pr, y_gt):
        jaccard = super().forward(y_pr, y_gt)
        bce = self.bce(y_pr, y_gt)
        return jaccard + bce


class BCEDiceLoss(DiceLoss):
    __name__ = 'bce_dice_loss'

    def __init__(self, eps=1e-7, activation='sigmoid'):
        super().__init__(eps, activation)
        self.bce = nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self, y_pr, y_gt):
        dice = super().forward(y_pr, y_gt)
        bce = self.bce(y_pr, y_gt)
        return dice + bce

class WeightedBCEDiceLoss(DiceLoss):
    __name__ = 'weighted_bce_dice_loss'

    def __init__(self, eps=1e-7, activation='sigmoid'):
        super().__init__(eps, activation)

    def criterion_pixel(self,logit_pixel, truth_pixel):
        logit = logit_pixel.view(-1)
        truth = truth_pixel.view(-1)
        assert(logit.shape==truth.shape)

        loss = nn.functional.binary_cross_entropy_with_logits(logit, truth, reduction='none')
        
        pos = (torch.sigmoid(truth)>0.5).float()
        neg = (torch.sigmoid(truth)<0.5).float()
        pos_weight = pos.sum().item() + 1e-12
        neg_weight = neg.sum().item() + 1e-12
        loss = (0.25*pos*loss/pos_weight + 0.75*neg*loss/neg_weight).sum()

        return loss

    def forward(self, y_pr, y_gt):
        dice = super().forward(y_pr, y_gt)
        # bce = self.bce(y_pr, y_gt)
        bce = self.criterion_pixel(y_pr, y_gt)
        return dice + bce

class WeightedBCELoss(nn.Module):
    __name__ = 'weighted_bce_loss'

    def __init__(self, eps=1e-7, activation='sigmoid'):
        super().__init__()

    def criterion_pixel(self,logit_pixel, truth_pixel):
        logit = logit_pixel.view(-1)
        truth = truth_pixel.view(-1)
        assert(logit.shape==truth.shape)

        loss = nn.functional.binary_cross_entropy_with_logits(logit, truth, reduction='none')
        
        pos = (truth>0.5).float()
        neg = (truth<0.5).float()
        pos_weight = pos.sum().item() + 1e-12
        neg_weight = neg.sum().item() + 1e-12
        loss = (0.99*pos*loss/pos_weight + 0.01*neg*loss/neg_weight).sum()

        return loss

    def forward(self, y_pr, y_gt):
        bce = self.criterion_pixel(y_pr, y_gt)
        return bce

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2, activation='sigmoid', reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.activation = activation
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.activation == 'sigmoid':
            BCE_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduce=None)
        else:
            BCE_loss = nn.functional.binary_cross_entropy(inputs, targets, reduction=None)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

class FocalDiceLoss(nn.Module):

    __name__ = 'focal_dice_loss'

    def __init__(self, eps=1e-7, activation='sigmoid'):
        super().__init__()

        self.dice = DiceLoss(activation=activation)
        self.fl = FocalLoss(activation=activation)

    def forward(self, y_pr, y_gt):
        fl = self.fl(y_pr, y_gt)
        dice = self.dice(y_pr, y_gt)
        return dice + fl