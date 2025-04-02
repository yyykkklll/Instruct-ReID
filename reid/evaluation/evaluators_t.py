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
    def evaluate(self, query_loader, gallery_loader, query, gallery):
        query_features, query_labels = self.extract_features(self.model, query_loader)
        gallery_features, gallery_labels = self.extract_features(self.model, gallery_loader)

        features = OrderedDict()
        features.update(query_features)
        features.update(gallery_features)

        labels = OrderedDict()
        labels.update(query_labels)
        labels.update(gallery_labels)

        distmat = self.pairwise_distance(features, query, gallery)
        return self.eval(distmat, query, gallery)

    @staticmethod
    def extract_features(model, data_loader, print_freq=50):
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
                outputs = model(imgs, captions)
                _, _, _, fused_feats = outputs
                feats = fused_feats

                if feats is None:
                    raise ValueError("fused_feats is None, check model output")

                print(
                    f"Batch {i}: fused_feats mean={feats.mean().item():.4f}, std={feats.std().item():.4f}, min={feats.min().item():.4f}, max={feats.max().item():.4f}")

                batch_data = data_loader.dataset.data[i * data_loader.batch_size:(i + 1) * data_loader.batch_size]
                for (img_path, _, pid, _), feat in zip(batch_data, feats):
                    fname = img_path
                    features[fname] = feat.cpu()
                    labels[fname] = pid

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
        x = torch.cat([features[f[0]].unsqueeze(0) for f in query], 0)
        y = torch.cat([features[f[0]].unsqueeze(0) for f in gallery], 0)
        x = torch.nn.functional.normalize(x, p=2, dim=1)  # L2 归一化
        y = torch.nn.functional.normalize(y, p=2, dim=1)  # L2 归一化
        m, n = x.size(0), y.size(0)

        # 计算欧氏距离
        dist_m = torch.pow(x - y.t(), 2).sum(dim=1, keepdim=True).expand(m, n)

        # 调试：计算匹配和非匹配样本的距离统计
        query_ids = np.array([items[2] for items in query])
        gallery_ids = np.array([items[2] for items in gallery])
        matches = (query_ids[:, np.newaxis] == gallery_ids[np.newaxis, :]).astype(np.int32)
        dist_m_np = dist_m.cpu().numpy()

        match_dists = dist_m_np[matches == 1]
        non_match_dists = dist_m_np[matches == 0]

        print(
            f"Match distances: mean={match_dists.mean():.4f}, std={match_dists.std():.4f}, min={match_dists.min():.4f}, max={match_dists.max():.4f}")
        print(
            f"Non-match distances: mean={non_match_dists.mean():.4f}, std={non_match_dists.std():.4f}, min={non_match_dists.min():.4f}, max={non_match_dists.max():.4f}")

        return dist_m

    @staticmethod
    def eval(distmat, query, gallery):
        distmat = to_numpy(distmat)

        query_ids = np.array([items[2] for items in query])
        gallery_ids = np.array([items[2] for items in gallery])
        query_cams = np.array([items[3] for items in query])
        gallery_cams = np.array([items[3] for items in gallery])

        print(f"Query IDs: {len(set(query_ids))} unique IDs")
        print(f"Gallery IDs: {len(set(gallery_ids))} unique IDs")
        print(f"Intersection between query and gallery IDs: {len(set(query_ids).intersection(set(gallery_ids)))}")

        cmc_scores, mAP = Evaluator_t2i.eval_func(distmat, q_pids=query_ids, g_pids=gallery_ids,
                                                  q_camids=query_cams, g_camids=gallery_cams, max_rank=50)

        print("=" * 80)
        print('Mean AP: {:4.1%}'.format(mAP))
        print('CMC Scores:')
        cmc_topk = (1, 5, 10, 20, 50)
        for k in cmc_topk:
            print('  top-{:<4}{:12.1%}'.format(k, cmc_scores[k - 1]))
        return {'mAP': mAP, 'rank1': cmc_scores[0], 'rank5': cmc_scores[4], 'rank10': cmc_scores[9]}

    @staticmethod
    def eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
        num_q, num_g = distmat.shape
        if num_g < max_rank:
            max_rank = num_g
            print("Note: number of gallery samples is quite small, got {}".format(num_g))

        indices = np.argsort(distmat, axis=1)
        matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

        print(f"Matches shape: {matches.shape}, sum of matches: {matches.sum()}")

        all_cmc = []
        all_AP = []
        num_valid_q = 0

        for q_idx in range(num_q):
            q_pid = q_pids[q_idx]
            order = indices[q_idx]
            orig_cmc = matches[q_idx]

            if not np.any(orig_cmc):
                print(f"Query {q_idx} (pid={q_pid}) has no matches in gallery")
                all_AP.append(0.0)
                all_cmc.append(np.zeros(max_rank, dtype=np.int32))
                continue

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

            print(f"Query {q_idx} (pid={q_pid}): AP={AP:.4f}, num_rel={num_rel}")

        print(f"Number of valid queries (with matches): {num_valid_q}")
        print(f"Total queries: {num_q}")

        assert len(all_AP) == num_q, f"Expected {num_q} APs, but got {len(all_AP)}"
        assert len(all_cmc) == num_q, f"Expected {num_q} CMCs, but got {len(all_cmc)}"

        all_cmc = np.asarray(all_cmc).astype(np.float32)
        all_cmc = all_cmc.sum(0) / num_q  # 使用总查询数平均
        mAP = np.mean(all_AP)

        return all_cmc, mAP
