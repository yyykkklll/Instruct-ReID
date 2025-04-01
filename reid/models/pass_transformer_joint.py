from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer, ViTModel


class T2IReIDModel(nn.Module):
    def __init__(self, num_classes, net_config):
        super(T2IReIDModel, self).__init__()
        self.net_config = net_config

        # Instruction Encoder: BERT
        bert_base_path = net_config.bert_base_path
        self.tokenizer = BertTokenizer.from_pretrained(bert_base_path)
        self.text_encoder = BertModel.from_pretrained(bert_base_path)
        self.text_width = self.text_encoder.config.hidden_size  # 768

        # Visual Transformer: ViT
        img_size = getattr(net_config, 'img_size', (224, 112))
        self.visual_encoder = ViTModel.from_pretrained("google/vit-base-patch16-224")
        self.in_planes = 768  # ViT base 的嵌入维度

        # 融合模块
        self.fusion = nn.MultiheadAttention(embed_dim=self.in_planes, num_heads=8)
        self.feat_dim = 512  # 输出特征维度
        self.fusion_proj = nn.Linear(self.in_planes, self.feat_dim)

        # 分类器
        self.num_classes = num_classes
        self.classifier = nn.Linear(self.feat_dim, num_classes, bias=False)
        nn.init.normal_(self.classifier.weight, std=0.001)

        # 队列机制（可选，保留以支持对比学习）
        self.queue_size = 4096
        self.register_buffer("image_queue", torch.randn(self.feat_dim, self.queue_size))
        self.register_buffer("text_queue", torch.randn(self.feat_dim, self.queue_size))
        self.register_buffer("idx_queue", torch.full((1, self.queue_size), -100))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.image_queue = nn.functional.normalize(self.image_queue, dim=0)
        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)

    def forward(self, image=None, instruction=None, label=None, task_info=None):
        device = next(self.parameters()).device

        # 图像特征 (Visual Transformer)
        if image is not None:
            image_outputs = self.visual_encoder(image)
            image_embeds = image_outputs.last_hidden_state[:, 0, :]  # [batch, 768], [CLS] token
        else:
            image_embeds = None

        # 文本特征 (BERT)
        if instruction is not None:
            instruction_text = self.tokenizer(instruction, padding='max_length', max_length=70,
                                              return_tensors="pt").to(device)
            text_outputs = self.text_encoder(**instruction_text)
            text_embeds = text_outputs.last_hidden_state[:, 0, :]  # [batch, 768], [CLS] token
        else:
            text_embeds = None

        # 融合特征
        if image_embeds is not None and text_embeds is not None:
            image_embeds = image_embeds.unsqueeze(0)  # [1, batch, 768]
            text_embeds = text_embeds.unsqueeze(0)  # [1, batch, 768]
            fused_feats, _ = self.fusion(image_embeds, text_embeds, text_embeds)
            fused_feats = fused_feats.squeeze(0)  # [batch, 768]
            fused_feats = self.fusion_proj(fused_feats)  # [batch, 512]
        else:
            fused_feats = None

        # 损失计算
        loss_dict = {}
        if self.training and fused_feats is not None and label is not None:
            logits = self.classifier(fused_feats)
            loss_dict['ce_loss'] = F.cross_entropy(logits, label)

            # 队列更新（可选）
            image_feats = F.normalize(fused_feats, dim=-1)
            text_feats = F.normalize(fused_feats, dim=-1)  # 这里简化，实际可单独投影
            idx = label
            self._dequeue_and_enqueue(image_feats, text_feats, idx)

        return loss_dict

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path, map_location='cpu', weights_only=True)
        loaded_layers = 0
        total_layers = len(param_dict)

        for key, value in param_dict.items():
            key = key.replace('module.', '')
            if key in self.state_dict() and self.state_dict()[key].shape == value.shape:
                self.state_dict()[key].copy_(value)
                loaded_layers += 1

        print(f'Loaded {loaded_layers} / {total_layers} layers from {trained_path}')

    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat, text_feat, idx):
        if torch.distributed.is_initialized():
            image_feats = concat_all_gather(image_feat)
            text_feats = concat_all_gather(text_feat)
            idxs = concat_all_gather(idx)
        else:
            image_feats = image_feat
            text_feats = text_feat
            idxs = idx

        batch_size = image_feats.shape[0]
        ptr = int(self.queue_ptr)
        empty = self.image_queue.size(1) - ptr
        if batch_size <= empty:
            self.image_queue[:, ptr:ptr + batch_size] = image_feats.T
            self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
            self.idx_queue[:, ptr:ptr + batch_size] = idxs.T
        else:
            self.image_queue[:, ptr:] = image_feats[:empty].T
            self.text_queue[:, ptr:] = text_feats[:empty].T
            self.idx_queue[:, ptr:] = idxs[:empty].T
            self.image_queue[:, :batch_size - empty] = image_feats[empty:].T
            self.text_queue[:, :batch_size - empty] = text_feats[empty:].T
            self.idx_queue[:, :batch_size - empty] = idxs[empty:].T
        ptr = (ptr + batch_size) % self.image_queue.size(1)
        self.queue_ptr[0] = ptr


@torch.no_grad()
def concat_all_gather(tensor):
    if not torch.distributed.is_initialized():
        return tensor
    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    return torch.cat(tensors_gather, dim=0)
