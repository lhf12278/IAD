from utils.logger import setup_logger
from datasets import make_dataloader
from model import make_model
from solver import make_optimizer1, make_optimizer2,make_optimizer_Attack,WarmupMultiStepLR
from solver.scheduler_factory import create_scheduler,create_scheduler_attack
from loss import make_loss
from processor import do_train
import random
import torch
import numpy as np
import os
import os.path as osp
import argparse
from config import cfg
import torch.distributed as dist
from model.My_Model import DG_Net1,Attack
import torch.nn as nn
train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

E1, E2 = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
DG_Net = DG_Net1(E1,num_classes=num_classes)
DG_Net.E2.load_state_dict(torch.load(cfg.TEST.WEIGHT), strict=False)

device = "cuda"


if device:
    if torch.cuda.device_count() > 1:
        print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
        model = nn.DataParallel(DG_Net)
    model.to(device)

model.eval()

for key, value in val_loader.items():
    num_q = num_query[key]
    for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(value):
        with torch.no_grad():
            img = img.to(device)
            # targets = vid.to(device)
            feat = model(img)
            # evaluator.update((feat, vid, camid))