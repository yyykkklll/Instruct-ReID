from __future__ import print_function, absolute_import

import time
from collections import OrderedDict

import numpy as np
import torch
from reid.utils import to_torch, to_numpy
from reid.utils.meters import AverageMeter


class Evaluator_t2i(object):
    def __init__(self, model):
        super(Evaluator_t2i, self).__init__()
        self.model = model

    @torch.no_grad()
    def evaluate(self, data_loader, query, gallery):
        """Evaluate the model on the T2I-ReID task."""
        features = self.extract_features(self.model, data_loader)
        distmat = self.pairwise_distance(features, query, gallery)
        return self.eval(distmat, query, gallery)

    @staticmethod
    def extract_features(model, data_loader, print_freq=50):
        """Extract features from the model for all samples in the data_loader."""
        model.eval()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        features = OrderedDict()
        labels = OrderedDict()

        end = time.time()
        with torch.no_grad():
            for i, data in enumerate(data_loader):
                imgs, captions, pids, cam_ids = data
                data_time.update(time.time() - end)

                imgs = to_torch(imgs).cuda()
                feats = model(imgs, captions)  # T2IReIDModel 返回融合特征

                for fname, pid, feat in zip([f"{i}_{pid.item()}" for i in range(len(pids))], pids, feats):
                    features[fname] = feat.cpu()
                    labels[fname] = pid.item()

                batch_time.update(time.time() - end)
                end = time.time()

                if (i + 1) % print_freq == 0:
                    print('Extract Features: [{}/{}]\t'
                          'Time {:.3f} ({:.3f})\t'
                          'Data {:.3f} ({:.3f})\t'
                          .format(i + 1, len(data_loader),
                                  batch_time.val, batch_time.avg,
                                  data_time.val, data_time.avg))

        return features, labels

    @staticmethod
    def pairwise_distance(features, query, gallery):
        """Compute pairwise distance between query and gallery features."""
        x = torch.cat([features[f[0]].unsqueeze(0) for f in query], 0)
        y = torch.cat([features[f[0]].unsqueeze(0) for f in gallery], 0)
        m, n = x.size(0), y.size(0)
        x = x.view(m, -1)
        y = y.view(n, -1)
        dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                 torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        dist_m.addmm_(1, -2, x, y.t())
        return dist_m

    @staticmethod
    def eval(distmat, query, gallery):
        """Evaluate mAP and CMC scores."""
        distmat = to_numpy(distmat)

        query_ids = [items[2] for items in query]
        gallery_ids = [items[2] for items in gallery]
        query_cams = [items[3] for items in query]
        gallery_cams = [items[3] for items in gallery]

        query_ids = np.asarray(query_ids)
        gallery_ids = np.asarray(gallery_ids)
        query_cams = np.asarray(query_cams)
        gallery_cams = np.asarray(gallery_cams)

        cmc_scores, mAP = Evaluator_t2i.eval_func(distmat, q_pids=query_ids, g_pids=gallery_ids,
                                                  q_camids=query_cams, g_camids=gallery_cams, max_rank=50)

        print("=" * 80)
        print('Mean AP: {:4.1%}'.format(mAP))
        print('CMC Scores:')
        cmc_topk = (1, 5, 10, 20, 50)
        for k in cmc_topk:
            print('  top-{:<4}{:12.1%}'.format(k, cmc_scores[k - 1]))
        return mAP

    @staticmethod
    def eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
        """Evaluation with market1501-like metric."""
        num_q, num_g = distmat.shape
        if num_g < max_rank:
            max_rank = num_g
            print("Note: number of gallery samples is quite small, got {}".format(num_g))

        indices = np.argsort(distmat, axis=1)
        matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

        all_cmc = []
        all_AP = []
        num_valid_q = 0

        for q_idx in range(num_q):
            q_pid = q_pids[q_idx]
            q_camid = q_camids[q_idx]

            order = indices[q_idx]
            remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
            keep = np.invert(remove)

            if not np.any(matches[q_idx][keep]):
                continue

            orig_cmc = matches[q_idx][keep]
            cmc = orig_cmc.cumsum()
            cmc[cmc > 1] = 1

            all_cmc.append(cmc[:max_rank])
            num_valid_q += 1

            num_rel = orig_cmc.sum()
            tmp_cmc = orig_cmc.cumsum()
            tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
            tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
            AP = tmp_cmc.sum() / max(1, num_rel)
            all_AP.append(AP)

        assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

        all_cmc = np.asarray(all_cmc).astype(np.float32)
        all_cmc = all_cmc.sum(0) / num_valid_q
        mAP = np.mean(all_AP)

        return all_cmc, mAP
