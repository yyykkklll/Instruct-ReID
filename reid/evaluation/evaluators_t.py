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
        """Evaluate the model on query (text) and gallery (image) data."""
        query_features, query_labels = self.extract_text_features(query_loader)
        gallery_features, gallery_labels = self.extract_image_features(gallery_loader)

        distmat = self.pairwise_distance(query_features, gallery_features)
        return self.eval(distmat, query, gallery)

    def extract_text_features(self, data_loader):
        """Extract text features from query data."""
        self.model.eval()
        features = OrderedDict()
        labels = OrderedDict()
        time_meter = AverageMeter()

        with torch.no_grad():
            for i, data in enumerate(data_loader):
                start_time = time.time()
                imgs, captions, pids, cam_ids = data
                imgs = to_torch(imgs).cuda()
                _, text_feats = self.model(imgs, captions)  # T2I-ReID: text features as query

                # 边界检查
                start_idx = i * data_loader.batch_size
                end_idx = min((i + 1) * data_loader.batch_size, len(data_loader.dataset.data))
                batch_data = data_loader.dataset.data[start_idx:end_idx]

                for (img_path, _, pid, _), feat in zip(batch_data, text_feats):
                    features[img_path] = feat.cpu()
                    labels[img_path] = pid

                time_meter.update(time.time() - start_time)

        total_time = time_meter.sum
        print(f"Text Feature Extraction: {len(features)} queries processed, "
              f"Total Time: {total_time:.2f}s, Avg Time per Query: {total_time / len(features):.4f}s")
        return features, labels

    def extract_image_features(self, data_loader):
        """Extract image features from gallery data."""
        self.model.eval()
        features = OrderedDict()
        labels = OrderedDict()
        time_meter = AverageMeter()

        with torch.no_grad():
            for i, data in enumerate(data_loader):
                start_time = time.time()
                imgs, captions, pids, cam_ids = data
                imgs = to_torch(imgs).cuda()
                image_feats, _ = self.model(imgs, captions)  # T2I-ReID: image features as gallery

                # 边界检查
                start_idx = i * data_loader.batch_size
                end_idx = min((i + 1) * data_loader.batch_size, len(data_loader.dataset.data))
                batch_data = data_loader.dataset.data[start_idx:end_idx]

                for (img_path, _, pid, _), feat in zip(batch_data, image_feats):
                    features[img_path] = feat.cpu()
                    labels[img_path] = pid

                time_meter.update(time.time() - start_time)

        total_time = time_meter.sum
        print(f"Image Feature Extraction: {len(features)} gallery items processed, "
              f"Total Time: {total_time:.2f}s, Avg Time per Item: {total_time / len(features):.4f}s")
        return features, labels

    @staticmethod
    def pairwise_distance(query_features, gallery_features):
        """Compute pairwise L2 distance between query and gallery features."""
        x = torch.cat([feat.unsqueeze(0) for fname, feat in query_features.items()], 0)
        y = torch.cat([feat.unsqueeze(0) for fname, feat in gallery_features.items()], 0)
        x = torch.nn.functional.normalize(x, p=2, dim=1)  # L2 normalization
        y = torch.nn.functional.normalize(y, p=2, dim=1)  # L2 normalization

        distmat = torch.cdist(x, y, p=2)  # L2 distance
        return distmat

    @staticmethod
    def eval(distmat, query, gallery):
        """Evaluate mAP and CMC scores."""
        distmat = to_numpy(distmat)
        query_ids = np.array([items[2] for items in query])  # pid as label
        gallery_ids = np.array([items[2] for items in gallery])

        cmc_scores, mAP = Evaluator_t2i.eval_func(distmat, q_pids=query_ids, g_pids=gallery_ids)

        # 详细打印结果
        print("=" * 80)
        print("Evaluation Results:")
        print(f"  Number of Queries: {len(query_ids)}")
        print(f"  Number of Gallery Items: {len(gallery_ids)}")
        print(f"  Mean AP (mAP): {mAP:.4f} ({mAP:.1%})")
        print("  CMC Scores:")
        cmc_topk = (1, 5, 10)
        for k in cmc_topk:
            print(f"    Rank-{k:<2}: {cmc_scores[k - 1]:.4f} ({cmc_scores[k - 1]:.1%})")
        print("=" * 80)

        return {'mAP': mAP, 'rank1': cmc_scores[0], 'rank5': cmc_scores[4], 'rank10': cmc_scores[9]}

    @staticmethod
    def eval_func(distmat, q_pids, g_pids, max_rank=10):
        """Compute CMC and mAP metrics."""
        num_q, num_g = distmat.shape
        if num_g < max_rank:
            max_rank = num_g
            print(f"Note: Gallery size ({num_g}) smaller than max_rank, adjusted to {max_rank}")

        indices = np.argsort(distmat, axis=1)  # Sort distances
        matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

        all_cmc = []
        all_AP = []
        num_valid_q = 0

        for q_idx in range(num_q):
            q_pid = q_pids[q_idx]
            order = indices[q_idx]
            orig_cmc = matches[q_idx]

            if not np.any(orig_cmc):  # No match
                all_AP.append(0.0)
                all_cmc.append(np.zeros(max_rank, dtype=np.int32))
                continue

            cmc = orig_cmc.cumsum()
            cmc[cmc > 1] = 1  # Cumulative match characteristic

            all_cmc.append(cmc[:max_rank])
            num_valid_q += 1

            num_rel = orig_cmc.sum()  # Number of relevant items
            tmp_cmc = orig_cmc.cumsum()
            tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]  # Precision
            tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
            AP = tmp_cmc.sum() / max(1, num_rel)  # Average precision
            all_AP.append(AP)

        all_cmc = np.asarray(all_cmc).astype(np.float32)
        all_cmc = all_cmc.sum(0) / num_q  # Average CMC across queries
        mAP = np.mean(all_AP)  # Mean AP

        return all_cmc, mAP
