import os
import torch
from reid.loss.adv_loss import InfoNCELoss, CosFaceLoss


class T2IReIDTrainer:
    def __init__(self, model, task_info):
        self.model = model
        self.task_info = task_info
        self.args = task_info
        self.scaler = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.info_nce_loss = InfoNCELoss(temperature=0.07).to(self.device)
        self.cosface_loss = CosFaceLoss(scale=16, margin=0.1).to(self.device)

    def run(self, inputs, task_info=None):
        image, caption, pid, cam_id = inputs

        image = image.to(self.device)
        pid = pid.to(self.device)
        cam_id = cam_id.to(self.device) if cam_id is not None else None

        # 前向传播
        with torch.amp.autocast('cuda', enabled=self.args.fp16):
            loss_dict, image_feats, text_feats, fused_feats = self.model(image=image, instruction=caption, label=pid,
                                                                         task_info=self.task_info)

            # 确保 image_feats 和 text_feats 是二维张量
            if image_feats is not None:
                while image_feats.dim() > 2:
                    image_feats = image_feats.squeeze()
            if text_feats is not None:
                while text_feats.dim() > 2:
                    text_feats = text_feats.squeeze()

            # 添加 InfoNCE 损失
            if image_feats is not None and text_feats is not None:
                # 确保形状为 [batch_size, feat_dim]
                if image_feats.shape[0] != self.args.batch_size:
                    image_feats = image_feats.mT  # 从 [feat_dim, batch_size] 转置为 [batch_size, feat_dim]
                if text_feats.shape[0] != self.args.batch_size:
                    text_feats = text_feats.mT  # 从 [feat_dim, batch_size] 转置为 [batch_size, feat_dim]
                loss_dict['info_nce_loss'] = self.info_nce_loss(image_feats, text_feats)

            # 可选：添加 CosFace 损失
            if fused_feats is not None:
                logits = self.model.classifier(fused_feats)
                loss_dict['cosface_loss'] = self.cosface_loss(logits, pid)

        return loss_dict

    def train(self, train_loader, optimizer, lr_scheduler, test_loader=None, query=None, gallery=None):
        self.model.train()
        for epoch in range(1, self.args.epochs + 1):
            total_loss = 0
            for i, inputs in enumerate(train_loader):
                optimizer.zero_grad()
                loss_dict = self.run(inputs)
                loss = sum(loss_dict.values()) * 0.5  # 平均 InfoNCE 和 CosFace 损失

                if self.args.fp16:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                total_loss += loss.item()

                if i % self.args.print_freq == 0:
                    avg_loss = total_loss / (i + 1)
                    print(f"Epoch {epoch}, Batch {i}/{len(train_loader)}, Loss: {avg_loss:.4f}, "
                          f"LR: {optimizer.param_groups[0]['lr']:.6f}")

            lr_scheduler.step()

            if epoch % self.args.save_freq == 0:
                save_path = os.path.join(self.args.logs_dir, f"checkpoint_epoch_{epoch}.pth")
                torch.save({
                    'model': self.model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch
                }, save_path)
                print(f"Model saved at: {save_path}")
