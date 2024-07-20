import logging
import os
import time
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from torch.cuda import amp
import torch.distributed as dist
import torch.nn.functional as F
from model.My_Model import cos_simility_loss,load_model_with_max_number,compute_class_centers,update_memory
import copy

def find_inf_nan_indices(loss_tensor):
    nan_indices = torch.isnan(loss_tensor).nonzero().flatten().tolist()
    inf_indices = torch.isinf(loss_tensor).nonzero().flatten().tolist()
    if nan_indices or inf_indices:
        return nan_indices, inf_indices
    else:
        return None



def do_train(cfg,
             DG_Net,
             Attack_Net,
             train_loader,
             val_loader,
             optimizer_DG_Net,
             optimizer_Attack_Net,
             scheduler_DG_Net,
             scheduler_Attack_Net,
             loss_fn,
             num_query,local_rank,output_dir,train_transforms1):
    log_period = cfg.SOLVER.LOG_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = "cuda"

    logger = logging.getLogger("transreid.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    if device:
        DG_Net.to(local_rank)
        Attack_Net.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            DG_Net = torch.nn.parallel.DistributedDataParallel(DG_Net, device_ids=[local_rank],
                                                                find_unused_parameters=True)
            Attack_Net = torch.nn.parallel.DistributedDataParallel(Attack_Net, device_ids=[local_rank],
                                                                   find_unused_parameters=True)

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    scaler = amp.GradScaler()
    epochs = cfg.SOLVER.MAX_EPOCHS

    a = cfg.PARA.TRAIN_DG_Net  # A显示的次数
    b = cfg.PARA.TRAIN_Attack_Net

    current_a = 0
    current_b = 0

    s_a = 1
    s_b = 1
    best_map = 0.0
    best_rank1 = 0.0
    best_map_epoch = 0
    best_rank1_epoch =0
    for epoch in range(1, epochs + 1):

        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()

        if current_a == a and current_b == b:
            current_a = 0
            current_b = 0
        if current_a < a:
            module = 0
            module_epoch = s_a
            s_a += 1
            current_a += 1

        else:
            if current_b < b:
                module = 1
                module_epoch = s_b
                s_b += 1
                current_b += 1
            elif current_a == a or current_b == b:
                current_a = 0
                current_b = 0


        if module == 0:
            DG_Net.train()
            Attack_Net.eval()
        elif module == 1:
            DG_Net.train()
            Attack_Net.train()

        for n_iter, (img, vid, target_cam, target_view,img_path) in enumerate(train_loader):

            optimizer_DG_Net.zero_grad()
            optimizer_Attack_Net.zero_grad()


            if module == 1:
                half_batch = cfg.SOLVER.IMS_PER_BATCH // 2
                img1 = img[:half_batch].to(device)
                target1 = vid[:half_batch].to(device)

                img2 = img[half_batch:].to(device)
                target2 = vid[half_batch:].to(device)
                imgs = [img1, img2]
                targets = [target1, target2]

            img = img.to(device)
            target = vid.to(device)

            with amp.autocast(enabled=True):

                if module == 0:
                    with torch.no_grad():
                        x_noise = Attack_Net(img)
                        x_noise = train_transforms1(x_noise.detach())
                        img = train_transforms1(img)
                    feat_list, score_list, kl_loss = DG_Net(img, x_noise)  ### ,part_loss,part_score_list

                    loss = loss_fn(score_list, feat_list, target) + kl_loss


                elif module == 1:

                    Attack_model= load_model_with_max_number(output_dir)
                    Attack_Net1 = copy.deepcopy(Attack_Net)
                    Attack_Net1.load_state_dict(Attack_model, strict=False)
                    Attack_Net1.eval()

                    for index, img in enumerate(imgs):
                        target = targets[index]
                        align_loss = nn.MSELoss(reduction='mean')
                        x_noise = Attack_Net(img)
                        x_noise_old = Attack_Net1(img).detach()
                        img = train_transforms1(img)
                        x_noise1 = train_transforms1(x_noise)

                        _, score_list, _ = DG_Net(img, x_noise1)

                        global_score1 = score_list[1]

                        ali_loss = align_loss(x_noise,  x_noise_old)

                        loss_average = F.cross_entropy(global_score1, target)

                        loss = (1-cfg.PARA.ASG_LANBUDA)*ali_loss-cfg.PARA.ASG_LANBUDA * loss_average
                        loss.backward()
                        optimizer_Attack_Net.step()
                        torch.cuda.empty_cache()
                    target = targets[1]

            if module == 0:
                scaler.scale(loss).backward()
                scaler.step(optimizer_DG_Net)
                scaler.update()

            if isinstance(score_list, list):
                acc = (score_list[1].max(1)[1] == target).float().mean()
            else:
                acc = (score_list.max(1)[1] == target).float().mean()

            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            if module == 0:
                if (n_iter + 1) % log_period == 0:
                    logger.info("DG_Net :Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                                .format(epoch, (n_iter + 1), len(train_loader),
                                        loss_meter.avg, acc_meter.avg, scheduler_DG_Net._get_lr(epoch)[0]))

            elif module == 1:
                if (n_iter + 1) % log_period == 0:
                    logger.info("Attack_Net :Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                                .format(epoch, (n_iter + 1), len(train_loader),
                                        loss_meter.avg, acc_meter.avg, scheduler_Attack_Net._get_lr(module_epoch)[0]))


        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)

        if module == 0:
            scheduler_DG_Net.step(module_epoch)
        elif module == 1:
            scheduler_Attack_Net.step(module_epoch)

        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                        .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))

        if module == 0:
            torch.save(Attack_Net.state_dict(), os.path.join(output_dir, 'Attack' + '_{}.pth'.format(module_epoch)))
            if module_epoch % eval_period == 0:
                DG_Net.eval()
                for key, value in val_loader.items():
                    num_q = num_query[key]
                    evaluator = R1_mAP_eval(num_q, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
                    evaluator.reset()
                    for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(value):
                        with torch.no_grad():
                            img = img.to(device)
                            feat = DG_Net(img ,img)
                            evaluator.update((feat, vid, camid))
                    cmc, mAP, _, _, _, _, _ = evaluator.compute()

                    if key != cfg.DATASETS.NAMES[0]:

                        if mAP > best_map:
                            torch.save(DG_Net.E.state_dict(),
                                       os.path.join(output_dir,
                                                    cfg.MODEL.NAME + '_best_map.pth'))
                            best_map = mAP
                            best_map_epoch = module_epoch

                        if cmc[0]>best_rank1:
                            torch.save(DG_Net.E.state_dict(),
                                       os.path.join(output_dir,
                                                    cfg.MODEL.NAME + '_best_rank1.pth'))
                            best_rank1 = cmc[0]
                            best_rank1_epoch = module_epoch

                    logger.info("Validation Results----{}---{}".format(key, module_epoch))
                    logger.info("mAP: {:.1%}---------best_mAP:{:.1%}------best_epoch:{}".format(mAP,best_map,best_map_epoch))
                    logger.info("Rank-1: {:.1%}------best_Rank-1:{:.1%}---best_epoch:{}".format(cmc[0], best_rank1,best_rank1_epoch))

                    for r in [5, 10]:
                        logger.info("Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))

                torch.cuda.empty_cache()



def do_inference(cfg,
                 model,
                 val_loader,
                 num_query):
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")


    model = nn.DataParallel(model)
    model.to(device)

    model.eval()

    for key, value in val_loader.items():
        num_q = num_query[key]
        evaluator = R1_mAP_eval(num_q, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
        evaluator.reset()
        for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(value):
            with torch.no_grad():
                img = img.to(device)
                # targets = vid.to(device)
                feat = model(img, img)
                evaluator.update((feat, vid, camid))
        cmc, mAP, _, _, _, _, _ = evaluator.compute()


        logger.info("Validation Results----{}".format(key))
        logger.info("mAP: {:.1%}".format(mAP))

        for r in [1, 5, 10]:
            logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))


