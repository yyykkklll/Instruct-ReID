import time
import numpy as np
import torch
import logging
from pathlib import Path
from ..utils import to_torch, to_numpy


class Evaluator_t2i:
    """
    文本到图像检索评估器，用于计算 mAP 和 CMC 指标。
    """
    def __init__(self, model, args=None):
        """
        初始化评估器。

        Args:
            model: 待评估的模型
            args: 配置参数
        """
        self.model = model
        self.args = args
        self.device = next(model.parameters()).device
        self.gallery_features = None
        self.gallery_labels = None

    @torch.no_grad()
    def evaluate(self, query_loader, gallery_loader, query, gallery, checkpoint_path=None, epoch=None):
        """
        执行评估流程。

        Args:
            query_loader: 查询数据加载器
            gallery_loader: 图库数据加载器
            query: 查询数据集
            gallery: 图库数据集
            checkpoint_path: 模型检查点路径
            epoch: 当前训练轮次

        Returns:
            dict: 包含 mAP 和 CMC 指标的字典
        """
        start_time = time.time()
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(checkpoint.get('model', checkpoint), strict=False)

        self.model.eval()
        if self.gallery_features is None or self.gallery_labels is None:
            self.gallery_features, self.gallery_labels = self.extract_features(gallery_loader)
        query_features, query_labels = self.extract_features(query_loader)
        distmat = self.pairwise_distance(query_features, self.gallery_features)
        metrics = self.eval(distmat, query, gallery)

        if epoch is not None:
            logging.info(f"Epoch {epoch}: mAP={metrics['mAP']:.4f}, Rank-1={metrics['rank1']:.4f}")
        logging.info(f"Evaluation time: {time.time() - start_time:.2f}s")
        return metrics

    def extract_features(self, data_loader):
        """
        提取特征向量（优化：批量存储归一化特征）。

        Args:
            data_loader: 数据加载器

        Returns:
            tuple: (features, labels)，特征张量和标签列表
        """
        self.model.eval()
        features = []
        labels = []
        gate_image_means = []
        gate_text_means = []
        start_time = time.time()

        with torch.no_grad():
            for i, data in enumerate(data_loader):
                imgs, cloth_captions, id_captions, pids, _, _ = data
                imgs = to_torch(imgs).to(self.device)

                try:
                    outputs = self.model(imgs, cloth_instruction=cloth_captions, id_instruction=id_captions)
                    fused_feats, gate_weights = outputs[2], outputs[-1]
                except AttributeError:
                    logging.error("Model does not support fused feature extraction")
                    raise

                # 归一化特征（优化：提前归一化）
                fused_feats = torch.nn.functional.normalize(fused_feats, p=2, dim=1)
                features.append(fused_feats.cpu())

                labels.extend(pids.tolist())

                # 记录 gate_weights 统计（优化：直接计算均值）
                if gate_weights is not None:
                    gate_image_means.append(gate_weights[:, 0].mean().item())
                    gate_text_means.append(gate_weights[:, 1].mean().item())

        # 合并特征（优化：一次性 cat）
        features = torch.cat(features, dim=0)
        labels = np.array(labels)

        # 计算 gate_weights 统计（优化：一次性计算均值和标准差）
        if gate_image_means and gate_text_means:
            image_mean = sum(gate_image_means) / len(gate_image_means)
            text_mean = sum(gate_text_means) / len(gate_text_means)
            image_std = (sum((x - image_mean) ** 2 for x in gate_image_means) / len(gate_image_means)) ** 0.5
            text_std = (sum((x - text_mean) ** 2 for x in gate_text_means) / len(gate_text_means)) ** 0.5
            logging.debug(f"Gate weights: Image mean={image_mean:.4f}, std={image_std:.4f}; "
                          f"Text mean={text_mean:.4f}, std={text_std:.4f}")

        logging.debug(f"Feature extraction time: {time.time() - start_time:.2f}s")
        return features, labels

    def pairwise_distance(self, query_features, gallery_features):
        """
        计算查询特征与图库特征之间的距离矩阵（优化：使用归一化特征）。

        Args:
            query_features: 查询特征张量
            gallery_features: 图库特征张量

        Returns:
            torch.Tensor: 距离矩阵
        """
        similarities = torch.matmul(query_features, gallery_features.t())
        distmat = 2 - 2 * similarities
        return distmat

    def eval(self, distmat, query, gallery):
        """
        计算评估指标。

        Args:
            distmat: 距离矩阵
            query: 查询数据集
            gallery: 图库数据集

        Returns:
            dict: 包含 mAP 和 CMC 指标的字典
        """
        distmat = to_numpy(distmat)
        query_ids = np.array([items[3] for items in query])
        gallery_ids = np.array([items[3] for items in gallery])
        cmc_scores, mAP = self.eval_func(distmat, query_ids, gallery_ids)

        # 限制指标范围（优化：直接在返回时限制）
        return {
            'mAP': min(mAP, 1.0),
            'rank1': min(cmc_scores[0], 1.0),
            'rank5': min(cmc_scores[4], 1.0),
            'rank10': min(cmc_scores[9], 1.0)
        }

    @staticmethod
    def eval_func(distmat, q_pids, g_pids, max_rank=10):
        """
        计算 CMC 和 mAP 指标的核心函数（优化：减少循环开销）。

        Args:
            distmat: 距离矩阵
            q_pids: 查询 ID 列表
            g_pids: 图库 ID 列表
            max_rank: 最大排名

        Returns:
            tuple: (cmc_scores, mAP)，CMC 得分和 mAP 值
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
            match = matches[q_idx]
            if not match.any():
                continue

            cmc = match.cumsum()
            cmc[cmc > 1] = 1
            all_cmc.append(cmc[:max_rank])

            num_rel = match.sum()
            tmp_cmc = match.cumsum() / (np.arange(len(match)) + 1)
            tmp_cmc *= match
            AP = tmp_cmc.sum() / max(1, num_rel)
            all_AP.append(AP)
            num_valid_q += 1

        if num_valid_q == 0:
            return np.zeros(max_rank, dtype=np.float32), 0.0

        all_cmc = np.asarray(all_cmc).mean(axis=0)
        mAP = np.mean(all_AP)
        return all_cmc, mAP