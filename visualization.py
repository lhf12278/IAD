import numpy as np
import os.path as osp
import os
from PIL import Image, ImageOps, ImageDraw
import torch.nn as nn
from os.path import join, realpath, dirname
import os
from config import cfg
import argparse
from utils.metrics import R1_mAP_eval
from datasets import make_dataloader
from model import make_model
from processor import do_inference
from utils.logger import setup_logger
from model.My_Model import DG_Net,Attack
import torch


def make_dirs(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
        print('Successfully make dirs: {}'.format(dir))
    else:
        print('Existed dirs: {}'.format(dir))


def hamming_distance(x, y):
    """
    compute hamming distance (NOT hamming similarity)
    between x and y
    Args:
        x(np.ndarray): [num_x, dim] in {0, 1}
        y(np.ndarray): [num_y, dim] in {0, 1}
    Return:
        (np.ndarray): [num_x, num_y]
    """
    assert min(x.min(), y.min())==0 and max(x.max(), y.max())==1, \
        'expect binary codes in \{0, 1\}, but got {{}, {}}'.format(min(x.min(), y.min()), max(x.max(), y.max()))

    assert x.shape[1] == y.shape[1], \
        'expect x and y have the same dimmension, but got x {} and y {}'.format(x.shape[1], y.shape[1])
    code_len = x.shape[1]

    x = (x-0.5)*2
    y = (y-0.5)*2
    return code_len - (np.matmul(x, y.transpose([1,0])) + code_len) / 2

def visualize_ranked_results(distmat, dataset, save_dir='./vis-results/', sort='ascend', topk=20, mode='inter-camera',
                             show='all', display_score=False):
    """Visualizes ranked results.
    Args:
        dismat (numpy.ndarray): distance matrix of shape (nq, ng)
        dataset (tupple): a 2-tuple including (query,gallery), each of which contains
            tuples of (img_paths, pids, camids)
        save_dir (str): directory to save output images.
        topk (int, optional): denoting top-k images in the rank list to be visualized.
        sort (string): ascend means small value is similar, otherwise descend
        mode (string): intra-camera/inter-camera/all
            intra-camera only visualize results in the same camera with the query
            inter-camera only visualize results in the different camera with the query
            all visualize all results
        show(string): pos/neg/all
            pos onlu show those true matched images
            neg only show those false matched images
            all show both
    """
    num_q, num_g = distmat.shape

    print('Visualizing top-{} ranks'.format(topk))
    print('# query: {}\n# gallery {}'.format(num_q, num_g))
    print('Saving images to "{}"'.format(save_dir))

    query, gallery = dataset
    assert num_q == len(query)
    assert num_g == len(gallery)
    assert sort in ['ascend', 'descend']
    assert mode in ['intra-camera', 'inter-camera', 'all']
    assert show in ['pos', 'neg', 'all']

    if sort == 'ascend':
        indices = np.argsort(distmat, axis=1)
    else:
        indices = np.argsort(distmat, axis=1)[:, ::-1]
    os.makedirs(save_dir, exist_ok=True)

    def cat_imgs_to(image_list, hit_list, text_list, target_dir):

        images = []
        for img, hit, text in zip(image_list, hit_list, text_list):
            img = Image.open(img).resize((64, 128))
            d = ImageDraw.Draw(img)
            if display_score:
                d.text((3, 1), "{:.3}".format(text), fill=(255, 255, 0))
            if hit:
                img = ImageOps.expand(img, border=4, fill='green')
            else:
                img = ImageOps.expand(img, border=4, fill='red')
            images.append(img)

        widths, heights = zip(*(i.size for i in images))
        total_width = sum(widths)
        max_height = max(heights)
        new_im = Image.new('RGB', (total_width, max_height))
        x_offset = 0
        for im in images:
            new_im.paste(im, (x_offset, 0))
            x_offset += im.size[0]

        new_im.save(target_dir)

    counts = 0
    for q_idx in range(num_q):

        image_list = []
        hit_list = []
        text_list = []

        # query image
        qimg_path, qpid, qcamid = query[q_idx]
        image_list.append(qimg_path)
        hit_list.append(True)
        text_list.append(0.0)

        # target dir
        if isinstance(qimg_path, tuple) or isinstance(qimg_path, list):
            qdir = osp.join(save_dir, osp.basename(qimg_path[0]))
        else:
            qdir = osp.join(save_dir, osp.basename(qimg_path))

        # matched images
        rank_idx = 1
        for ii, g_idx in enumerate(indices[q_idx, :]):
            gimg_path, gpid, gcamid = gallery[g_idx]
            if mode == 'intra-camera':
                valid = qcamid == gcamid
            elif mode == 'inter-camera':
                valid = (qpid != gpid and qcamid == gcamid) or (qcamid != gcamid)
            elif mode == 'all':
                valid = True
            if valid:
                if show == 'pos' and qpid != gpid: continue
                if show == 'neg' and qpid == gpid: continue
                image_list.append(gimg_path)
                hit_list.append(qpid == gpid)
                text_list.append(distmat[q_idx, g_idx])
                rank_idx += 1
                if rank_idx > topk:
                    break

        counts += 1
        cat_imgs_to(image_list, hit_list, text_list, qdir)
        print(counts, qdir)



class CatMeter:
    '''
    Concatenate Meter for torch.Tensor
    '''
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = None

    def update(self, val):
        if self.val is None:
            self.val = val
        else:
            self.val = torch.cat([self.val, val], dim=0)
    def get_val(self):
        return self.val

    def get_val_numpy(self):
        return self.val.data.cpu().numpy()


class AverageMeter:
    """
    Average Meter
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.count = 0

    def update(self, val):
        self.sum += val
        self.count += 1

    def get_val(self):
        return self.sum / self.count



import time

if __name__ == "__main__":
    import sklearn.metrics.pairwise as skp
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="./configs/market/vit_small_ics.yml", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    save_path = cfg.VISUALIZA_DIR
    make_dirs(save_path)

    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

    # model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num = view_num)

    E1, E2 = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
    DG_Net = DG_Net(E1, E2, num_classes=num_classes)
    DG_Net.E2.load_state_dict(torch.load(cfg.TEST.WEIGHT), strict=False)
    DG_Net.E1.load_state_dict(torch.load(cfg.TEST.WEIGHT), strict=False)

    device = "cuda"

    model = nn.DataParallel(DG_Net)
    model.to(device)

    model.eval()


    for key, value in val_loader.items():
        num_q = num_query[key]
        evaluator = R1_mAP_eval(num_q, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
        evaluator.reset()
        features_meter = None
        pids_meter, camids_meter = CatMeter(), CatMeter()
        time_meter = AverageMeter()

        for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(value):
            with torch.no_grad():
                img = img.to(device)
                feat = model(img, img)

            #     if isinstance(feats, torch.Tensor):
            #         if features_meter is None:
            #             features_meter = CatMeter()
            #         features_meter.update(feats.data)
            #     elif isinstance(feats, list):
            #         if features_meter is None:
            #             features_meter = [CatMeter() for _ in range(len(feats))]
            #         for idx, feats_i in enumerate(feats):
            #             features_meter[idx].update(feats_i.data)
            #     else:
            #         assert 0
            #     pids_meter.update(pids.data)
            #     camids_meter.update(vid.data)
            #
            # if isinstance(features_meter, list):
            #     feats = [val.get_val_numpy() for val in features_meter]
            # else:
            #     feats = features_meter.get_val_numpy()
            # pids = pids_meter.get_val_numpy()
            # camids = camids_meter.get_val_numpy()

        _, _, _, pids, camids, qf, gf= evaluator.compute()

        distmat = skp.cosine_distances(qf, gf)
        sort = 'ascend'

        dataset = [value[:num_q],value[num_q:]]

        output_path = join(save_path, '{}'.format(key))

        visualize_ranked_results(distmat, dataset, save_dir=output_path, sort=sort,topk=20, mode='inter-camera', show='all')





