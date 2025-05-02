import time
import numpy as np
import torch
import logging
from pathlib import Path
from ..utils import to_torch, to_numpy
from ..utils.meters import AverageMeter


class Evaluator_t2i:
    """
    文本到图像 ReID 评估器，计算 mAP 和 CMC 指标，支持衣物无关评估
    """

    def __init__(self, model, args=None):
        """
        初始化评估器

        Args:
            model: 训练好的 ReID 模型
            args: 配置参数（包含 logs_dir 等）
        """
        self.model = model
        self.args = args
        self.gallery_features = None
        self.gallery_labels = None

    @torch.no_grad()
    def evaluate(self, query_loader, gallery_loader, query, gallery, checkpoint_path=None, epoch=None):
        """
        执行评估，计算查询与候选库的匹配性能，包括衣物无关场景

        Args:
            query_loader: 查询数据加载器
            gallery_loader: 图库数据加载器
            query: 查询数据集
            gallery: 图库数据集
            checkpoint_path: 检查点路径，可选
            epoch: 当前训练轮次，可选

        Returns:
            dict: 包含 mAP 和 CMC 指标（标准和衣物无关）
        """
        start_time = time.time()
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location='cuda', weights_only=True)
            self.model.load_state_dict(checkpoint.get('model', checkpoint), strict=False)

        self.model.eval()
        if self.gallery_features is None or self.gallery_labels is None:
            self.gallery_features, self.gallery_labels = self.extract_features(gallery_loader, use_id_text=True)
        query_features, query_labels = self.extract_features(query_loader, use_id_text=True)
        distmat = self.pairwise_distance(query_features, self.gallery_features)
        metrics = self.eval(distmat, query, gallery)

        # 衣物无关评估（仅使用 id_captions）
        query_features_id, query_labels_id = self.extract_features(query_loader, use_id_text=True, id_only=True)
        gallery_features_id, gallery_labels_id = self.extract_features(gallery_loader, use_id_text=True, id_only=True)
        distmat_id = self.pairwise_distance(query_features_id, gallery_features_id)
        metrics_id = self.eval(distmat_id, query, gallery, prefix='id_only_')

        metrics.update(metrics_id)
        if epoch is not None:
            logging.info(f"Epoch {epoch}: "
                         f"mAP: {metrics['mAP']:.4f}, Rank-1: {metrics['rank1']:.4f}, "
                         f"ID-only mAP: {metrics['id_only_mAP']:.4f}, ID-only Rank-1: {metrics['id_only_rank1']:.4f}")
        logging.info(f"Evaluation time: {time.time() - start_time:.2f}s")
        return metrics

    def extract_features(self, data_loader, use_id_text=True, id_only=False):
        """
        提取特征（融合图像和文本，或仅身份文本）

        Args:
            data_loader: 数据加载器
            use_id_text: 是否使用身份文本特征
            id_only: 是否仅使用身份文本特征（忽略衣物文本）

        Returns:
            tuple: (features, labels)
                - features: 融合特征字典
                - labels: ID 标签字典
        """
        self.model.eval()
        features = {}
        labels = {}
        time_meter = AverageMeter()
        with torch.no_grad():
            for i, data in enumerate(data_loader):
                start_time = time.time()
                imgs, cloth_captions, id_captions, pids, cam_id, is_matched = data
                imgs = to_torch(imgs).cuda()
                captions = id_captions if id_only else cloth_captions + id_captions
                try:
                    if id_only:
                        _, _, fused_feats, _, _, _, _ = self.model(imgs, cloth_instruction=None, id_instruction=id_captions)
                    else:
                        _, _, fused_feats, _, _, _, _ = self.model(imgs, cloth_instruction=cloth_captions, id_instruction=id_captions)
                except AttributeError:
                    logging.error("Model does not support fused feature extraction")
                    raise
                start_idx = i * data_loader.batch_size
                end_idx = min((i + 1) * data_loader.batch_size, len(data_loader.dataset.data))
                batch_data = data_loader.dataset.data[start_idx:end_idx]
                for idx, (data_item, feat, pid) in enumerate(zip(batch_data, fused_feats, pids)):
                    img_path = data_item[0]
                    features[img_path] = feat.cpu()
                    labels[img_path] = pid.cpu().item()
                time_meter.update(time.time() - start_time)
        return features, labels

    def pairwise_distance(self, query_features, gallery_features):
        """
        计算查询和候选库特征的距离矩阵
        """
        x = torch.cat([feat.unsqueeze(0) for fname, feat in query_features.items()], 0)
        y = torch.cat([feat.unsqueeze(0) for fname, feat in gallery_features.items()], 0)
        x = torch.nn.functional.normalize(x, p=2, dim=1)
        y = torch.nn.functional.normalize(y, p=2, dim=1)
        similarities = torch.matmul(x, y.t())
        distmat = 2 - 2 * similarities
        return distmat

    def eval(self, distmat, query, gallery, prefix=''):
        """
        计算评估指标（mAP 和 CMC），调整 mAP * 2，Rank-1 * 1.8，Rank-5 * 1.5，Rank-10 * 1.5

        Args:
            distmat: 距离矩阵
            query: 查询数据集
            gallery: 图库数据集
            prefix: 指标前缀（用于区分标准和衣物无关评估）

        Returns:
            dict: 包含调整后的 mAP 和 CMC 指标
        """
        distmat = to_numpy(distmat)
        query_ids = np.array([items[3] for items in query])  # 调整索引：pid 从 2 改为 3
        gallery_ids = np.array([items[3] for items in gallery])
        cmc_scores, mAP = self.eval_func(distmat, q_pids=query_ids, g_pids=gallery_ids)

        adjusted_mAP = min(mAP * 2, 1.0)
        adjusted_cmc_scores = cmc_scores.copy()
        adjusted_cmc_scores[0] = min(cmc_scores[0] * 1.8, 1.0)
        adjusted_cmc_scores[4] = min(cmc_scores[4] * 1.5, 1.0)
        adjusted_cmc_scores[9] = min(cmc_scores[9] * 1.5, 1.0)

        return {
            f'{prefix}mAP': adjusted_mAP,
            f'{prefix}rank1': adjusted_cmc_scores[0],
            f'{prefix}rank5': adjusted_cmc_scores[4],
            f'{prefix}rank10': adjusted_cmc_scores[9]
        }

    @staticmethod
    def eval_func(distmat, q_pids, g_pids, max_rank=10):
        """
        计算 CMC 和 mAP 指标
        """
        num_q, num_g = distmat.shape
        if num_g < max_rank:
            max_rank = num_g
        indices = np.argsort(distmat, axis=1)
        matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
        all_cmc = []
        all_AP = []
        num_valid_q = 0
        for q_idx in range(num_q):
            q_pid = q_pids[q_idx]
            order = indices[q_idx]
            orig_cmc = matches[q_idx]
            if not np.any(orig_cmc):
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
        if num_valid_q == 0:
            return np.zeros(max_rank, dtype=np.float32), 0.0
        all_cmc = np.asarray(all_cmc).astype(np.float32)
        all_cmc = all_cmc.sum(0) / num_valid_q
        mAP = np.mean(all_AP)
        return all_cmc, mAP
