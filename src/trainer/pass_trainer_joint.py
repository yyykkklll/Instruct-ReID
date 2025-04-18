import os
from pathlib import Path
import torch

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
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.args.fp16)

    def run(self, inputs):
        image, caption, pid, cam_id = inputs
        image = image.to(self.device)
        pid = pid.to(self.device)
        cam_id = cam_id.to(self.device) if cam_id is not None else None

        # 前向传播
        with torch.amp.autocast('cuda', enabled=self.args.fp16):
            image_feats, text_feats = self.model(image=image, instruction=caption)

            # 计算损失
            loss_dict = self.combined_loss(image_feats, text_feats, pid)

        return loss_dict

    def train(self, train_loader, optimizer, lr_scheduler, query_loader=None, query=None, gallery=None):
        self.model.train()
        best_mAP = 0.0
        best_checkpoint = None

        # 项目根目录定位
        ROOT_DIR = Path(__file__).parent.parent.parent  # 从 src/trainer/ 到 TextGuidedReID/

        for epoch in range(1, self.args.epochs + 1):
            total_loss = 0
            total_info_nce = 0
            total_clip = 0
            total_triplet = 0
            batch_count = 0

            for i, inputs in enumerate(train_loader):
                optimizer.zero_grad()
                loss_dict = self.run(inputs)
                loss = loss_dict['total_loss']

                if self.args.fp16:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                total_loss += loss.item()
                total_info_nce += loss_dict['info_nce_loss'].item()
                total_clip += loss_dict['clip_loss'].item()
                total_triplet += loss_dict['triplet_loss'].item()
                batch_count += 1

                if i % self.args.print_freq == 0:
                    avg_loss = total_loss / (i + 1)
                    avg_info_nce = total_info_nce / batch_count
                    avg_clip = total_clip / batch_count
                    avg_triplet = total_triplet / batch_count
                    print(f"Epoch {epoch}, Batch {i}/{len(train_loader)}, "
                          f"Loss: {avg_loss:.4f}, "
                          f"InfoNCE: {avg_info_nce:.4f}, "
                          f"CLIP: {avg_clip:.4f}, "
                          f"Triplet: {avg_triplet:.4f}, "
                          f"LR: {optimizer.param_groups[0]['lr']:.6f}")

            lr_scheduler.step()

            # 在 save_freq 时保存并评估
            if epoch % self.args.save_freq == 0 or epoch == self.args.epochs:
                save_path = os.path.join(ROOT_DIR, self.args.logs_dir, f"checkpoint_epoch_{epoch:03d}.pth")
                torch.save({
                    'model': self.model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch
                }, save_path)
                print(f"Model saved at: {save_path}")

                # 评估
                if query_loader is not None and query is not None and gallery is not None:
                    evaluator = Evaluator_t2i(self.model)
                    metrics = evaluator.evaluate(query_loader, query_loader, query, gallery)
                    print(f"Epoch {epoch} Evaluation Results:")
                    print(metrics)

                    # 更新最佳检查点
                    mAP = metrics['mAP']
                    if mAP > best_mAP:
                        best_mAP = mAP
                        best_checkpoint = save_path
                        print(f"New best checkpoint saved: {best_checkpoint} with mAP: {best_mAP:.4f}")

        # 打印最佳检查点
        if best_checkpoint:
            print("=" * 80)
            print(f"Training completed. Best checkpoint: {best_checkpoint} with mAP: {best_mAP:.4f}")
            print("=" * 80)