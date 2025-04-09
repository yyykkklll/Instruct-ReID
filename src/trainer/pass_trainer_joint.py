import os
from pathlib import Path
import torch
import logging

from ..loss.adv_loss import CombinedLoss
from ..evaluation.evaluators_t import Evaluator_t2i


class T2IReIDTrainer:
    def __init__(self, model, args):
        self.model = model
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.combined_loss = CombinedLoss(
            temperature_info=0.07,
            temperature_clip=0.1,
            margin_triplet=0.3,
            weights=(1.0, 1.0, 1.0)
        ).to(self.device)
        self.scaler = torch.amp.GradScaler('cuda', enabled=args.fp16)
        self.accum_steps = 2  # 梯度累积步数

    def run(self, inputs):
        image, caption, pid, cam_id = inputs
        image = image.to(self.device)
        pid = pid.to(self.device)
        cam_id = cam_id.to(self.device) if cam_id is not None else None

        image_feats, text_feats = self.model(image=image, instruction=caption)
        if image_feats is None or text_feats is None:
            logging.error(f"Invalid model output - Image feats: {image_feats}, Text feats: {text_feats}")
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        loss_dict = self.combined_loss(image_feats, text_feats, pid)
        total_loss = loss_dict['total_loss']
        return total_loss

    def train_epoch(self, train_loader, optimizer, lr_scheduler, epoch, scaler):
        self.model.train()
        total_loss = 0
        total_info_nce = 0
        total_clip = 0
        total_triplet = 0
        batch_count = 0

        optimizer.zero_grad()
        for i, inputs in enumerate(train_loader):
            with torch.amp.autocast('cuda', enabled=self.args.fp16):
                loss = self.run(inputs)

            if not isinstance(loss, torch.Tensor):
                logging.error(f"Epoch {epoch + 1}, Batch {i + 1}: Loss is not a tensor: {loss}")
                continue
            if not loss.requires_grad:
                logging.warning(f"Epoch {epoch + 1}, Batch {i + 1}: Loss does not require grad: {loss.item()}")
                continue

            # 梯度累积
            scaled_loss = scaler.scale(loss) / self.accum_steps
            scaled_loss.backward()

            if (i + 1) % self.accum_steps == 0 or (i + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            # 记录损失
            with torch.no_grad():
                image, caption, pid, _ = inputs
                image = image.to(self.device)
                pid = pid.to(self.device)
                image_feats, text_feats = self.model(image, caption)
                loss_dict = self.combined_loss(image_feats, text_feats, pid)
                total_loss += loss_dict['total_loss'].item()
                total_info_nce += loss_dict['info_nce_loss'].item()
                total_clip += loss_dict['clip_loss'].item()
                total_triplet += loss_dict['triplet_loss'].item()
            batch_count += 1

            if (i + 1) % self.args.print_freq == 0:
                avg_loss = total_loss / batch_count
                avg_info_nce = total_info_nce / batch_count
                avg_clip = total_clip / batch_count
                avg_triplet = total_triplet / batch_count
                log_msg = (f"Epoch {epoch + 1}, Batch {i + 1}/{len(train_loader)}, "
                           f"Loss: {avg_loss:.4f}, "
                           f"InfoNCE: {avg_info_nce:.4f}, "
                           f"CLIP: {avg_clip:.4f}, "
                           f"Triplet: {avg_triplet:.4f}, "
                           f"LR: {optimizer.param_groups[0]['lr']:.6f}")
                logging.info(log_msg)

        lr_scheduler.step()

    def train(self, train_loader, optimizer, lr_scheduler, query_loader=None, query=None, gallery=None):
        self.model.train()
        best_mAP = 0.0
        best_checkpoint = None
        ROOT_DIR = Path(__file__).parent.parent.parent

        for epoch in range(self.args.epochs):
            self.train_epoch(train_loader, optimizer, lr_scheduler, epoch, self.scaler)

            if (epoch + 1) % self.args.save_freq == 0 or (epoch + 1) == self.args.epochs:
                save_path = os.path.join(ROOT_DIR, self.args.logs_dir, f"checkpoint_epoch_{epoch + 1:03d}.pth")
                torch.save({
                    'model': self.model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch + 1
                }, save_path)
                logging.info(f"Model saved at: {save_path}")

                if query_loader is not None and query is not None and gallery is not None:
                    evaluator = Evaluator_t2i(self.model)
                    metrics = evaluator.evaluate(query_loader, query_loader, query, gallery)
                    logging.info(f"Epoch {epoch + 1} Evaluation Results:")
                    logging.info(f"  mAP: {metrics['mAP']:.4f}, Rank-1: {metrics['rank1']:.4f}, "
                                 f"Rank-5: {metrics['rank5']:.4f}, Rank-10: {metrics['rank10']:.4f}")

                    mAP = metrics['mAP']
                    if mAP > best_mAP:
                        best_mAP = mAP
                        best_checkpoint = save_path
                        best_save_path = os.path.join(ROOT_DIR, self.args.logs_dir, 'checkpoint_best.pth')
                        torch.save({
                            'model': self.model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'lr_scheduler': lr_scheduler.state_dict(),
                            'epoch': epoch + 1
                        }, best_save_path)
                        logging.info(f"New best checkpoint saved: {best_save_path} with mAP: {best_mAP:.4f}")

        if best_checkpoint:
            logging.info("=" * 80)
            logging.info(f"Training completed. Best checkpoint: {best_checkpoint} with mAP: {best_mAP:.4f}")
            logging.info("=" * 80)