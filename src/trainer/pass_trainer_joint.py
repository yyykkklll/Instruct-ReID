import logging
import torch
from pathlib import Path
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
from ..loss.adv_loss import AdvancedLoss
from ..evaluation.evaluators_t import Evaluator_t2i
from ..utils.serialization import save_checkpoint
from ..utils.meters import AverageMeter


class T2IReIDTrainer:
    def __init__(self, model, args):
        """
        初始化 T2I-ReID 训练器。

        Args:
            model: T2I-ReID 模型
            args: 命令行参数
        """
        self.model = model
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        loss_weights = args.disentangle.get('loss_weights', {
            'info_nce': 1.0, 'cls': 1.0, 'bio': 0.5, 'cloth': 0.5, 'cloth_adv': 0.1,
            'cloth_match': 1.0, 'decouple': 0.1, 'gate_regularization': 0.01
        })
        self.combined_loss = AdvancedLoss(temperature=0.1, weights=loss_weights).to(self.device)
        self.scaler = GradScaler(enabled=args.fp16) if self.device.type == 'cuda' else None
        if args.fp16 and self.device.type != 'cuda':
            logging.warning("FP16 enabled but no CUDA device available. Disabling mixed precision.")

    def run(self, inputs, epoch, batch_idx, total_batches):
        """
        运行单次训练迭代。

        Args:
            inputs: 数据批次 (image, cloth_captions, id_captions, pid, cam_id, is_matched)
            epoch (int): 当前 epoch
            batch_idx (int): 当前批次索引
            total_batches (int): 总批次数

        Returns:
            dict: 损失字典
        """
        image, cloth_captions, id_captions, pid, cam_id, is_matched = inputs
        image = image.to(self.device)
        pid = pid.to(self.device)
        cam_id = cam_id.to(self.device) if cam_id is not None else None
        is_matched = is_matched.to(self.device)

        with autocast(enabled=self.args.fp16):
            outputs = self.model(image=image, cloth_instruction=cloth_captions, id_instruction=id_captions)
            if len(outputs) != 10:
                raise ValueError(f"Unexpected number of model outputs: {len(outputs)}, expected 10")
            image_feats, id_text_feats, fused_feats, id_logits, id_embeds, \
                cloth_embeds, cloth_text_embeds, cloth_image_embeds, gate, gate_weights = outputs

            loss_dict = self.combined_loss(
                image_embeds=image_feats, id_text_embeds=id_text_feats, fused_embeds=fused_feats,
                id_logits=id_logits, id_embeds=id_embeds, cloth_embeds=cloth_embeds,
                cloth_text_embeds=cloth_text_embeds, cloth_image_embeds=cloth_image_embeds,
                pids=pid, is_matched=is_matched, epoch=epoch, gate=gate
            )
            if gate_weights is not None and 'gate_weights_regularization' in self.combined_loss.weights:
                gate_weights_loss = torch.mean((gate_weights - 0.5) ** 2)
                loss_dict['gate_weights_regularization'] = gate_weights_loss
                loss_dict['total'] += self.combined_loss.weights['gate_weights_regularization'] * gate_weights_loss

        return loss_dict

    def compute_similarity(self, train_loader):
        """
        计算训练数据上的相似度。

        Args:
            train_loader: 训练数据加载器

        Returns:
            tuple: (pos_sim, neg_sim, None, scale)
        """
        self.model.eval()
        with torch.no_grad():
            for image, cloth_captions, id_captions, pid, cam_id, is_matched in train_loader:
                image = image.to(self.device)
                outputs = self.model(image=image, cloth_instruction=cloth_captions, id_instruction=id_captions)
                image_feats, id_text_feats, _, _, _, _, _, _, _, gate_weights = outputs
                sim = torch.matmul(image_feats, id_text_feats.t())
                pos_sim = sim.diag().mean().item()
                neg_sim = sim[~torch.eye(sim.shape[0], dtype=bool, device=self.device)].mean().item()
                scale = self.model.scale.item()
                self.model.train()
                return pos_sim, neg_sim, None, scale
        self.model.train()
        return None, None, None, None

    def train(self, train_loader, optimizer, lr_scheduler, query_loader=None, gallery_loader=None, checkpoint_dir=None):
        """
        训练模型。

        Args:
            train_loader: 训练数据加载器
            optimizer: 优化器
            lr_scheduler: 学习率调度器
            query_loader: 查询数据加载器
            gallery_loader: 图库数据加载器
            checkpoint_dir: 检查点保存目录
        """
        self.model.train()
        best_mAP = 0.0
        best_checkpoint = None
        total_batches = len(train_loader)
        weights = self.combined_loss.weights
        logging.info(f"Training with loss weights: {weights}")

        # 动态初始化 loss_meters
        loss_meters = {k: AverageMeter() for k in weights.keys()}
        loss_meters['total'] = AverageMeter()

        for epoch in range(1, self.args.epochs + 1):
            # 打印上一个 epoch 的平均损失
            if epoch > 1:
                avg_losses = {k: m.avg for k, m in loss_meters.items() if m.count > 0}
                if avg_losses:
                    logging.info(f"[Epoch {epoch-1}] Avg Losses: {', '.join(f'{k}={v:.4f}' for k, v in avg_losses.items())}")

            # 重置 meters
            for meter in loss_meters.values():
                meter.reset()

            with tqdm(
                train_loader,
                desc=f"[Epoch {epoch}/{self.args.epochs}] Training",
                ncols=140,
                total=total_batches,
                leave=True
            ) as progress_bar:
                for i, inputs in enumerate(progress_bar):
                    optimizer.zero_grad()
                    loss_dict = self.run(inputs, epoch, i, total_batches)
                    loss = loss_dict['total']

                    if self.scaler:
                        self.scaler.scale(loss).backward()
                        self.scaler.step(optimizer)
                        self.scaler.update()
                    else:
                        loss.backward()
                        optimizer.step()

                    # 更新 loss_meters
                    for key, value in loss_dict.items():
                        if key in loss_meters:
                            loss_meters[key].update(value if isinstance(value, float) else value.item())

            lr_scheduler.step()

            # 修改：每 10 个 epoch 或最终 epoch 保存检查点并评估
            if epoch % 10 == 0 or epoch == self.args.epochs:
                save_path = Path(checkpoint_dir) / f"checkpoint_epoch_{epoch:03d}.pth"
                save_checkpoint({
                    'model': self.model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch
                }, fpath=str(save_path))
                logging.info(f"Checkpoint saved at: {save_path}")

                # 评估
                if query_loader and gallery_loader:
                    with torch.no_grad():
                        evaluator = Evaluator_t2i(self.model, args=self.args)
                        metrics = evaluator.evaluate(
                            query_loader, gallery_loader,
                            query_loader.dataset.data, gallery_loader.dataset.data,
                            checkpoint_path=str(save_path), epoch=epoch
                        )
                    mAP = metrics['mAP']
                    logging.info(f"Evaluation at epoch {epoch}: mAP={mAP:.4f}, Metrics={metrics}")
                    if mAP > best_mAP:
                        best_mAP = mAP
                        best_checkpoint = save_path
                        logging.info(f"New best checkpoint: {best_checkpoint}, mAP={mAP:.4f}")

        # 打印最终平均损失
        avg_losses = {k: m.avg for k, m in loss_meters.items() if m.count > 0}
        if avg_losses:
            logging.info(f"[Epoch {epoch}] Avg Losses: {', '.join(f'{k}={v:.4f}' for k, v in avg_losses.items())}")

        if best_checkpoint:
            logging.info(f"Final best checkpoint: {best_checkpoint}, mAP={best_mAP:.4f}")