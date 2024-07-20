from utils.logger import setup_logger
from datasets import make_dataloader
from model import make_model
from solver import make_optimizer1,make_optimizer_Attack
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
from model.My_Model import DG_Net,Attack

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

if __name__ == '__main__':
    import setproctitle

    setproctitle.setproctitle("python")

    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="./configs/vit_small_ics.yml", help="path to config file", type=str
    )

    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument("--local_rank", default=0, type=int)
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    
    cfg.freeze()
    set_seed(cfg.SOLVER.SEED)

    if cfg.MODEL.DIST_TRAIN:
        torch.cuda.set_device(args.local_rank)

    if len(cfg.DATASETS.NAMES)==1:
        output_dir = osp.join(cfg.OUTPUT_DIR,
                              str(cfg.DATASETS.NAMES[0]) + '->' + str(cfg.DATASETS.NAMES_TARGET[0]) + '+' + str(
                                  cfg.DATASETS.NAMES_TARGET[1]))
    elif len(cfg.DATASETS.NAMES)==2:
        output_dir = osp.join(cfg.OUTPUT_DIR,
                              str(cfg.DATASETS.NAMES[0])+'+'+str(cfg.DATASETS.NAMES[1]) + '->' + str(cfg.DATASETS.NAMES_TARGET[0]))
    elif len(cfg.DATASETS.NAMES)==3:
        output_dir = osp.join(cfg.OUTPUT_DIR,
                              str(cfg.DATASETS.NAMES[0]) + '+' + str(cfg.DATASETS.NAMES[1])+ '+' + str(cfg.DATASETS.NAMES[2]) + '->' + str(
                                  cfg.DATASETS.NAMES_TARGET[0]))



    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)


    logger = setup_logger("transreid", output_dir, if_train=True)
    logger.info("Saving model in the path :{}".format(cfg.OUTPUT_DIR))
    #  logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            #  logger.info(config_str)

    if cfg.MODEL.DIST_TRAIN:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
    logger.info("Running with config:\n{}".format(cfg))


    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num,train_transforms1= make_dataloader(cfg)


    E = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num = view_num)
    DG_Net = DG_Net(E,num_classes=num_classes)
    loss_func = make_loss(cfg)
    Attack_Net = Attack(alpha=cfg.PARA.RATO, input_channel=3, output_channel=64, latent_dim=32)
    optimizer_DG_Net = make_optimizer1(cfg, DG_Net)
    optimizer_Attack_Net = make_optimizer_Attack(cfg, Attack_Net)


    scheduler_DG_Net = create_scheduler(cfg, optimizer_DG_Net)
    scheduler_Attack_Net = create_scheduler_attack(cfg, optimizer_Attack_Net)



    if cfg.SOLVER.WARMUP_METHOD == 'cosine':
        logger.info('===========using cosine learning rate=======')
        scheduler_DG_Net = create_scheduler(cfg, optimizer_DG_Net)
        scheduler_Attack_Net = create_scheduler_attack(cfg, optimizer_Attack_Net)

    do_train(
        cfg,
        DG_Net,
        Attack_Net,
        train_loader,
        val_loader,
        optimizer_DG_Net,
        optimizer_Attack_Net,
        scheduler_DG_Net,
        scheduler_Attack_Net,
        loss_func,
        num_query, args.local_rank,output_dir,train_transforms1
    )

