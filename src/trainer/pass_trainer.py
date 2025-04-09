from torch.nn import CrossEntropyLoss
import torch
import os

class T2ITrainer:
    def __init__(self, model, args, this_task_info=None, is_distributed=False):
        self.model = model
        self.args = args
        self.this_task_info = this_task_info
        self.is_distributed = is_distributed
        self.ce_loss = CrossEntropyLoss().cuda()

    def _parse_data(self, inputs):
        imgs, instructions, _, _, pids, view_ids, cam_ids, indices = inputs
        inputs = imgs.cuda()
        targets = pids.cuda()
        cam_ids = cam_ids.cuda()
        view_ids = view_ids.cuda()
        return inputs, instructions, targets, cam_ids, view_ids

    def run(self, inputs):
        inputs, instructions, targets, cam_ids, view_ids = self._parse_data(inputs)
        _, _, _, logits1, logits2, logits3, _ = self.model(
            inputs, instructions, this_task_info=self.this_task_info,
            cam_label=cam_ids, view_label=view_ids
        )
        loss = 0
        if isinstance(logits1, list):
            loss += sum(self.ce_loss(scor, targets) for scor in logits1) / len(logits1)
        else:
            loss += self.ce_loss(logits1, targets)
        return loss

    def train(self, train_loader, optimizer, lr_scheduler, test_loader=None, query=None, gallery=None):
        self.model.train()
        train_iter = iter(train_loader)  # 在循环前初始化 train_iter
        for cur_iter in range(1, self.args.iters + 1):
            try:
                inputs = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                inputs = next(train_iter)

            loss = self.run(inputs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            if cur_iter % self.args.print_freq == 0:
                print(f"Iter: [{cur_iter}/{self.args.iters}]\t"
                      f"Loss: {loss.item():.4f}\t"
                      f"LR: {optimizer.param_groups[0]['lr']:.6f}")

            if cur_iter % self.args.save_freq == 0:
                save_path = os.path.join(self.args.logs_dir, f"checkpoint_iter_{cur_iter}.pth")
                torch.save({
                    'model': self.model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'iter': cur_iter
                }, save_path)
                print(f"Model saved at: {save_path}")