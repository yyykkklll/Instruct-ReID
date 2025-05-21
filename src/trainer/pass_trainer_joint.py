import logging
import torch
from pathlib import Path
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
from ..loss.adv_loss import AdvancedLoss
from ..evaluation.evaluators_t import Evaluator_t2i
from ..utils.serialization import save_checkpoint
from ..utils.meters import AverageMeter


# class SilentStartTqdm(tqdm):
#     def display(self, msg=None, pos=None):
#         if self.n >= self.total:
#             super().display(msg, pos)


class T2IReIDTrainer:
    def __init__(self, model, args):
        self.model = model
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        loss_weights = args.disentangle.get('loss_weights', {
            'info_nce': 1.0, 'cls': 1.0, 'bio': 0.5, 'cloth': 0.5, 'cloth_adv': 0.1,
            'cloth_match': 1.0, 'decouple': 0.1, 'gate_regularization': 0.01
        })
        logging.info(f"使用损失权重配置: {loss_weights}")
        self.combined_loss = AdvancedLoss(temperature=0.1, weights=loss_weights).to(self.device)
        if self.device.type == 'cuda':
            self.scaler = GradScaler(enabled=args.fp16)
        else:
            self.scaler = None
            if args.fp16:
                logging.warning("FP16 is enabled but no CUDA device is available. Disabling mixed precision.")

    def run(self, inputs, epoch, batch_idx, total_batches):
        image, cloth_captions, id_captions, pid, cam_id, is_matched = inputs
        image = image.to(self.device)
        pid = pid.to(self.device)
        cam_id = cam_id.to(self.device) if cam_id is not None else None
        is_matched = is_matched.to(self.device)
        if batch_idx == 0:
            if not isinstance(cloth_captions, (list, tuple)) or not all(isinstance(c, str) for c in cloth_captions):
                raise ValueError("cloth_captions must be a list of strings")
            if not isinstance(id_captions, (list, tuple)) or not all(isinstance(c, str) for c in id_captions):
                raise ValueError("id_captions must be a list of strings")
            # logging.info(f"Sample cloth captions: {cloth_captions[:2]}")
            # logging.info(f"Sample id captions: {id_captions[:2]}")
            # print(f"Sample cloth captions: {cloth_captions[:2]}")
            # print(f"Sample ID captions: {id_captions[:2]}")
        with autocast(enabled=self.args.fp16):
            model_outputs = self.model(image=image, cloth_instruction=cloth_captions, id_instruction=id_captions)
            if len(model_outputs) == 10:
                image_feats, id_text_feats, fused_feats, id_logits, id_embeds, \
                cloth_embeds, cloth_text_embeds, cloth_image_embeds, gate, gate_weights = model_outputs
            else:
                raise ValueError(f"模型返回了未预期数量的输出: {len(model_outputs)}, 期望10个")
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
        self.model.eval()
        with torch.no_grad():
            for i, (image, cloth_captions, id_captions, pid, cam_id, is_matched) in enumerate(train_loader):
                if i == 0:
                    image = image.to(self.device)
                    outputs = self.model(image=image, cloth_instruction=cloth_captions, id_instruction=id_captions)
                    image_feats, id_text_feats, _, _, _, _, _, _, _, gate_weights = outputs
                    sim = torch.matmul(image_feats, id_text_feats.t())
                    pos_sim = sim.diag().mean().item()
                    neg_sim = sim[~torch.eye(sim.shape[0], dtype=bool, device=self.device)].mean().item()
                    scale = self.model.scale
                    if gate_weights is not None:
                        image_weight_mean = gate_weights[:, 0].mean().item()
                        text_weight_mean = gate_weights[:, 1].mean().item()
                        logging.info(f"Similarity computation - Gate weights: Image mean={image_weight_mean:.4f}, Text mean={text_weight_mean:.4f}")
                    return pos_sim, neg_sim, None, scale
        self.model.train()
        return None, None, None, None

    def train(self, train_loader, optimizer, lr_scheduler, query_loader=None, gallery_loader=None, checkpoint_dir=None):
        self.model.train()
        best_mAP = 0.0
        best_checkpoint = None
        total_batches = len(train_loader)
        
        weights = self.combined_loss.weights # 获取实际使用的损失权重
        print(f"训练使用的损失权重: {weights}")

        # 初始化 loss_meters 一次
        loss_meter_keys = list(self.combined_loss.weights.keys()) + ['total']
        loss_meters = {k: AverageMeter() for k in loss_meter_keys}

        for epoch in range(1, self.args.epochs + 1):
            # 从第二个 epoch 开始，在进度条之前打印上一个 epoch 的平均损失
            if epoch > 1:
                avg_loss_log_parts = []
                for key in weights.keys(): # 迭代实际的损失组件键
                    if key in loss_meters and loss_meters[key].count > 0:
                        avg_loss_log_parts.append(f"{key}={loss_meters[key].avg:.4f}")
                if avg_loss_log_parts: # 仅当有损失记录时打印
                    print(f"[Avg Loss:] : {', '.join(avg_loss_log_parts)}")
            
            # 为当前 epoch 重置 meters
            for meter in loss_meters.values():
                meter.reset()

            alpha = min(1.0, 2.0 * epoch / 50) # GRL alpha 或者其他用途的 alpha
            
            progress_bar = tqdm( 
                train_loader,
                desc=f"[Epoch {epoch}/{self.args.epochs}] Training",
                ncols=140,
                dynamic_ncols=True,
                leave=True, 
                total=total_batches
            )
            
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
                
                for key in loss_meters: # 更新当前 epoch 的 loss_meters
                    if key in loss_dict:
                        val = loss_dict[key].item() if isinstance(loss_dict[key], torch.Tensor) else loss_dict[key]
                        loss_meters[key].update(val)
                
            progress_bar.close()
            
            lr_scheduler.step()

            if epoch % self.args.save_freq == 0 or epoch == self.args.epochs:
                save_path = Path(checkpoint_dir) / f"checkpoint_epoch_{epoch:03d}.pth"
                save_checkpoint({
                    'model': self.model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch
                }, fpath=str(save_path))
                logging.info(f"Saved checkpoint at: {save_path}")

                if query_loader and gallery_loader:
                    evaluator = Evaluator_t2i(self.model, args=self.args)
                    metrics = evaluator.evaluate(
                        query_loader, gallery_loader,
                        query_loader.dataset.data, gallery_loader.dataset.data,
                        checkpoint_path=str(save_path), epoch=epoch
                    )
                    mAP = metrics['mAP']
                    if mAP > best_mAP:
                        best_mAP = mAP
                        best_checkpoint = save_path
                        logging.info(f"New best checkpoint: {best_checkpoint}, mAP: {best_mAP:.4f}")
        
        # 训练结束后，打印最后一个 epoch 的平均损失
        avg_loss_log_parts = []
        for key in weights.keys():
            if key in loss_meters and loss_meters[key].count > 0:
                avg_loss_log_parts.append(f"{key}={loss_meters[key].avg:.4f}")
        if avg_loss_log_parts:
            print(f"[Avg Loss:] : {', '.join(avg_loss_log_parts)}")

        if best_checkpoint:
            logging.info(f"Final best checkpoint: {best_checkpoint}, mAP: {best_mAP:.4f}")