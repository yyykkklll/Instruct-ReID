import logging
import torch
from pathlib import Path
from ..loss.adv_loss import AdvancedLoss
from ..evaluation.evaluators_t import Evaluator_t2i
from ..utils.serialization import save_checkpoint
from ..utils.meters import AverageMeter


class T2IReIDTrainer:
    """
    T2I-ReID 模型训练类，管理训练和评估流程，整合解纠缠损失
    """

    def __init__(self, model, args):
        """
        初始化训练器，设置模型、损失函数和混合精度训练

        Args:
            model: T2IReIDModel 实例
            args: 命令行参数 (Namespace)
        """
        self.model = model
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
<<<<<<< HEAD
        
        # 从配置中获取损失权重，如果未指定则使用默认值
        loss_weights = args.disentangle.get('loss_weights', {
            'info_nce': 1.0, 'cls': 0.5, 'bio': 0.2, 'cloth': 0.5, 'cloth_adv': 0.1,
            'cloth_match': 0.5, 'decouple': 0.3
        })
        
        # 记录使用的权重配置
        logging.info(f"使用损失权重配置: {loss_weights}")
        
        self.combined_loss = AdvancedLoss(
            temperature=0.1,
            weights=loss_weights
=======
        self.combined_loss = AdvancedLoss(
            temperature=0.07,
            weights=args.disentangle.get('loss_weights', {
                'info_nce': 1.0, 'cls': 0.5, 'bio': 0.1, 'cloth': 0.5, 'cloth_adv': 0.5
            })
>>>>>>> ae1d583f71d5b97df29d9414fb60417d2714e12b
        ).to(self.device)
        self.scaler = torch.amp.GradScaler('cuda', enabled=args.fp16)

    def run(self, inputs, epoch, batch_idx, total_batches):
        """
        执行单次前向传播并计算损失

        Args:
            inputs: 包含图像、衣物描述、身份描述、ID、相机ID和匹配标签的元组
            epoch: 当前轮次
            batch_idx: 当前批次索引
            total_batches: 总批次数

        Returns:
<<<<<<< HEAD
            dict: 损失字典，包含 total, info_nce, cls, bio, cloth, cloth_adv, cloth_match, decouple
=======
            dict: 损失字典，包含 total, info_nce, cls, bio, cloth, cloth_adv
>>>>>>> ae1d583f71d5b97df29d9414fb60417d2714e12b
        """
        # 解包6个字段
        image, cloth_captions, id_captions, pid, cam_id, is_matched = inputs
        image = image.to(self.device)
        pid = pid.to(self.device)
        cam_id = cam_id.to(self.device) if cam_id is not None else None
        is_matched = is_matched.to(self.device)

<<<<<<< HEAD
        # 验证输入格式（仅在第一批次）
=======
        # 验证输入格式
>>>>>>> ae1d583f71d5b97df29d9414fb60417d2714e12b
        if batch_idx == 0:
            if not isinstance(cloth_captions, (list, tuple)) or not all(isinstance(c, str) for c in cloth_captions):
                raise ValueError("cloth_captions must be a list of strings")
            if not isinstance(id_captions, (list, tuple)) or not all(isinstance(c, str) for c in id_captions):
                raise ValueError("id_captions must be a list of strings")
            logging.info(f"Sample cloth caption: {cloth_captions[:2]}")
            logging.info(f"Sample id caption: {id_captions[:2]}")

        with torch.amp.autocast('cuda', enabled=self.args.fp16):
            # 前向传播，获取解纠缠输出
<<<<<<< HEAD
            (image_feats, id_text_feats, fused_feats, id_logits, id_embeds,
             cloth_embeds, cloth_text_embeds, cloth_image_embeds) = self.model(
=======
            image_feats, id_text_feats, fused_feats, id_logits, id_embeds, cloth_embeds, cloth_text_embeds = self.model(
>>>>>>> ae1d583f71d5b97df29d9414fb60417d2714e12b
                image=image, cloth_instruction=cloth_captions, id_instruction=id_captions
            )

            # 计算损失
            loss_dict = self.combined_loss(
                image_embeds=image_feats,
                id_text_embeds=id_text_feats,
                fused_embeds=fused_feats,
                id_logits=id_logits,
                id_embeds=id_embeds,
                cloth_embeds=cloth_embeds,
                cloth_text_embeds=cloth_text_embeds,
<<<<<<< HEAD
                cloth_image_embeds=cloth_image_embeds,
                pids=pid,
                is_matched=is_matched,
                epoch=epoch
=======
                pids=pid
>>>>>>> ae1d583f71d5b97df29d9414fb60417d2714e12b
            )

        return loss_dict

    def compute_similarity(self, train_loader):
        """
        计算训练数据中正样本和负样本的特征相似度
<<<<<<< HEAD
=======

        Args:
            train_loader: 训练数据加载器

        Returns:
            tuple: (pos_sim, neg_sim, None, scale)
                - pos_sim: 正样本相似度均值
                - neg_sim: 负样本相似度均值
                - None: 占位符
                - scale: 模型缩放参数
>>>>>>> ae1d583f71d5b97df29d9414fb60417d2714e12b
        """
        self.model.eval()
        with torch.no_grad():
            for i, (image, cloth_captions, id_captions, pid, cam_id, is_matched) in enumerate(train_loader):
                if i == 0:
                    image = image.to(self.device)
                    image_feats, id_text_feats, _, _, _, _, _ = self.model(
                        image=image, cloth_instruction=cloth_captions, id_instruction=id_captions
                    )
                    sim = torch.matmul(image_feats, id_text_feats.t())
                    pos_sim = sim.diag().mean().item()
                    neg_sim = sim[~torch.eye(sim.shape[0], dtype=bool, device=self.device)].mean().item()
                    scale = self.model.scale
                    return pos_sim, neg_sim, None, scale
        self.model.train()
        return None, None, None, None

    def train(self, train_loader, optimizer, lr_scheduler, query_loader=None, gallery_loader=None, checkpoint_dir=None):
        """
        训练模型并定期评估，记录平均损失指标，每 args.print_freq 打印一次
<<<<<<< HEAD
=======

        Args:
            train_loader: 训练数据加载器
            optimizer: 优化器
            lr_scheduler: 学习率调度器
            query_loader: 查询数据加载器（评估用），可选
            gallery_loader: 图库数据加载器（评估用），可选
            checkpoint_dir: 检查点保存目录，可选
>>>>>>> ae1d583f71d5b97df29d9414fb60417d2714e12b
        """
        self.model.train()
        best_mAP = 0.0
        best_checkpoint = None
<<<<<<< HEAD
        total_batches = len(train_loader)
        
        # 记录损失权重配置
        weights = self.combined_loss.weights
        logging.info(f"训练使用的损失权重: info_nce={weights['info_nce']}, cls={weights['cls']}, "
                    f"bio={weights['bio']}, cloth={weights['cloth']}, cloth_adv={weights['cloth_adv']}, "
                    f"cloth_match={weights['cloth_match']}, decouple={weights['decouple']}")
        
        for epoch in range(1, self.args.epochs + 1):
            # 动态更新 GRL alpha - 修改为新的调度逻辑
            alpha = min(1.0, 2.0 * epoch / 50)  # 前50个epoch逐渐增加到1.0
            self.model.alpha = alpha
            logging.info(f"Epoch {epoch}/{self.args.epochs}: GRL alpha = {alpha:.4f}")

=======
        logger = logging.getLogger()
        total_batches = len(train_loader)

        for epoch in range(1, self.args.epochs + 1):
>>>>>>> ae1d583f71d5b97df29d9414fb60417d2714e12b
            # 初始化损失累计器
            loss_meters = {
                'total': AverageMeter(),
                'info_nce': AverageMeter(),
                'cls': AverageMeter(),
                'bio': AverageMeter(),
                'cloth': AverageMeter(),
<<<<<<< HEAD
                'cloth_adv': AverageMeter(),
                'cloth_match': AverageMeter(),
                'decouple': AverageMeter()
=======
                'cloth_adv': AverageMeter()
>>>>>>> ae1d583f71d5b97df29d9414fb60417d2714e12b
            }

            for i, inputs in enumerate(train_loader):
                optimizer.zero_grad()
                loss_dict = self.run(inputs, epoch, i, total_batches)
                loss = loss_dict['total']

                if self.args.fp16:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                # 累计损失
                for key in loss_meters:
                    value = loss_dict[key].item() if isinstance(loss_dict[key], torch.Tensor) else loss_dict[key]
                    loss_meters[key].update(value)

                # 每 args.print_freq 或最后一个batch打印平均损失
                if i % self.args.print_freq == 0 or i == total_batches - 1:
<<<<<<< HEAD
                    current_lr = optimizer.param_groups[0]['lr']
                    logging.info(
=======
                    logger.info(
>>>>>>> ae1d583f71d5b97df29d9414fb60417d2714e12b
                        f"Epoch {epoch}/{self.args.epochs}, Batch {i}/{total_batches}, "
                        f"Total: {loss_meters['total'].avg:.4f}, "
                        f"InfoNCE: {loss_meters['info_nce'].avg:.4f}, "
                        f"Cls: {loss_meters['cls'].avg:.4f}, "
                        f"Bio: {loss_meters['bio'].avg:.4f}, "
<<<<<<< HEAD
                        f"Clo: {loss_meters['cloth'].avg:.4f}, "
                        f"CloAdv: {loss_meters['cloth_adv'].avg:.4f}, "
                        f"CloMat: {loss_meters['cloth_match'].avg:.4f}, "
                        f"Dec: {loss_meters['decouple'].avg:.4f}, "
                        f"LR: {current_lr:.6f}"
=======
                        f"Cloth: {loss_meters['cloth'].avg:.4f}, "
                        f"ClothAdv: {loss_meters['cloth_adv'].avg:.4f}"
>>>>>>> ae1d583f71d5b97df29d9414fb60417d2714e12b
                    )

            # 在每个epoch结束后更新学习率调度器
            lr_scheduler.step()

            # 定期评估和保存
            if epoch % self.args.save_freq == 0 or epoch == self.args.epochs:
                save_path = Path(checkpoint_dir) / f"checkpoint_epoch_{epoch:03d}.pth"
                save_checkpoint({
                    'model': self.model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch
                }, fpath=str(save_path))
<<<<<<< HEAD
                logging.info(f"Saved checkpoint at: {save_path}")
=======
                logger.info(f"Saved: {save_path}")
>>>>>>> ae1d583f71d5b97df29d9414fb60417d2714e12b

                if query_loader is not None and gallery_loader is not None:
                    evaluator = Evaluator_t2i(self.model, args=self.args)
                    metrics = evaluator.evaluate(
                        query_loader, gallery_loader, query_loader.dataset.data, gallery_loader.dataset.data,
                        checkpoint_path=str(save_path), epoch=epoch
                    )
                    mAP = metrics['mAP']
                    if mAP > best_mAP:
                        best_mAP = mAP
                        best_checkpoint = save_path
<<<<<<< HEAD
                        logging.info(f"New best checkpoint: {best_checkpoint}, mAP: {best_mAP:.4f}")

        if best_checkpoint:
            logging.info(f"Final best checkpoint: {best_checkpoint}, mAP: {best_mAP:.4f}")
=======
                        logger.info(f"New best: {best_checkpoint}, mAP: {best_mAP:.4f}")

        if best_checkpoint:
            logger.info(f"Final: Best checkpoint: {best_checkpoint}, mAP: {best_mAP:.4f}")
>>>>>>> ae1d583f71d5b97df29d9414fb60417d2714e12b
