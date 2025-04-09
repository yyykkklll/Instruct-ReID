from __future__ import absolute_import

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, ViTModel
import os
import logging


class T2IReIDModel(nn.Module):
    def __init__(self, net_config):
        super(T2IReIDModel, self).__init__()
        self.net_config = net_config

        bert_base_path = net_config.bert_base_path
        vit_base_path = net_config.vit_pretrained

        if not os.path.exists(bert_base_path):
            raise FileNotFoundError(f"BERT路径未找到: {bert_base_path}")
        if not os.path.exists(vit_base_path):
            raise FileNotFoundError(f"ViT路径未找到: {vit_base_path}")

        self.tokenizer = BertTokenizer.from_pretrained(bert_base_path)
        self.text_encoder = BertModel.from_pretrained(bert_base_path)
        self.visual_encoder = ViTModel.from_pretrained(vit_base_path)
        self.feat_dim = 768

        self.modal_embedding = nn.Embedding(2, self.feat_dim)
        self.cross_modal_layer = nn.TransformerEncoderLayer(
            d_model=self.feat_dim, nhead=8, batch_first=True
        )

    def forward(self, image=None, instruction=None):
        device = next(self.parameters()).device

        # 图像特征提取
        image_embeds = None
        if image is not None:
            if image.dim() == 5:
                image = image.squeeze(-1)
            try:
                image_outputs = self.visual_encoder(image)
                image_tokens = image_outputs.last_hidden_state
                image_embeds = image_tokens.mean(dim=1)
            except Exception as e:
                logging.error(f"Image encoding failed: {e}")
                image_embeds = torch.zeros((image.size(0), self.feat_dim), device=device)

        # 文本特征提取
        text_embeds = None
        if instruction is not None:
            try:
                tokenized = self.tokenizer(
                    instruction,
                    padding='max_length',
                    max_length=100,  # 恢复到 100
                    truncation=True,
                    return_tensors="pt"
                ).to(device)
                text_outputs = self.text_encoder(**tokenized)
                text_tokens = text_outputs.last_hidden_state
                text_embeds = text_tokens.mean(dim=1)
            except Exception as e:
                logging.error(f"Text encoding failed: {e}")
                text_embeds = torch.zeros((len(instruction), self.feat_dim), device=device)

        # 跨模态融合
        if image_embeds is not None and text_embeds is not None:
            batch_size = image_tokens.size(0)
            image_modal = self.modal_embedding(
                torch.zeros(batch_size, image_tokens.size(1), dtype=torch.long).to(device)
            )
            text_modal = self.modal_embedding(
                torch.ones(batch_size, text_tokens.size(1), dtype=torch.long).to(device)
            )
            image_tokens = image_tokens + image_modal
            text_tokens = text_tokens + text_modal

            combined_tokens = torch.cat([image_tokens, text_tokens], dim=1)
            fused_tokens = self.cross_modal_layer(combined_tokens)
            num_img_tokens = image_tokens.size(1)
            fused_image_tokens = fused_tokens[:, :num_img_tokens, :]
            fused_text_tokens = fused_tokens[:, num_img_tokens:, :]
            image_embeds = fused_image_tokens.mean(dim=1)
            text_embeds = fused_text_tokens.mean(dim=1)

            # 清理中间变量
            del image_tokens, text_tokens, combined_tokens, fused_tokens, image_modal, text_modal
            torch.cuda.empty_cache()

        if image_embeds is not None:
            image_embeds = torch.nn.functional.normalize(image_embeds, dim=-1)
        if text_embeds is not None:
            text_embeds = torch.nn.functional.normalize(text_embeds, dim=-1)

        return image_embeds, text_embeds

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path, map_location='cpu', weights_only=True)
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']

        for key, value in param_dict.items():
            key = key.replace('module.', '')
            if key in self.state_dict() and self.state_dict()[key].shape == value.shape:
                self.state_dict()[key].copy_(value)