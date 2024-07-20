# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch.nn.functional as F
from .softmax_loss import CrossEntropyLabelSmooth, LabelSmoothingCrossEntropy
from .triplet_loss import TripletLoss
from .center_loss import CenterLoss


def make_loss(cfg):    # modified by gu

    triplet = TripletLoss()

    def loss_func(score, feat, target):
        if isinstance(score, list):
            ID_LOSS = [F.cross_entropy(scor, target) for scor in score[0:]]
            ID_LOSS = sum(ID_LOSS) / len(ID_LOSS)
        else:
            ID_LOSS = F.cross_entropy(score, target)
        if isinstance(feat, list):

            # TRI_LOSS = [triplet(feats,feats,feats, target,target,target) for feats in feat[0:]]
            # TRI_LOSS = sum(TRI_LOSS) / len(TRI_LOSS)
            TRI_LOSS = [triplet(feats, target, normalize_feature=cfg.SOLVER.TRP_L2)[0] for feats in feat[0:]]
            TRI_LOSS = sum(TRI_LOSS) / len(TRI_LOSS)

        else:
            TRI_LOSS = triplet(feat, target)[0]

        return cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS

    return loss_func


