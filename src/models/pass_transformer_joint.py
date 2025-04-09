from __future__ import absolute_import

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, ViTModel
from pathlib import Path
import os


class T2IReIDModel(nn.Module):
    def __init__(self, net_config):
        super(T2IReIDModel, self).__init__()
        self.net_config = net_config

        # 直接使用 net_config 中的绝对路径
        bert_base_path = net_config.bert_base_path
        vit_base_path = net_config.vit_pretrained

        # 检查路径是否存在
        if not os.path.exists(bert_base_path):
            raise FileNotFoundError(f"BERT base path not found at: {bert_base_path}")
        if not os.path.exists(vit_base_path):
            raise FileNotFoundError(f"ViT base path not found at: {vit_base_path}")

        self.tokenizer = BertTokenizer.from_pretrained(bert_base_path)
        self.text_encoder = BertModel.from_pretrained(bert_base_path)
        self.text_width = self.text_encoder.config.hidden_size  # 768

        self.visual_encoder = ViTModel.from_pretrained(vit_base_path)
        self.feat_dim = 768

    def forward(self, image=None, instruction=None):
        device = next(self.parameters()).device
        image_embeds = None
        if image is not None:
            if image.dim() == 5:
                image = image.squeeze(-1)
            image_outputs = self.visual_encoder(image)
            image_embeds = image_outputs.last_hidden_state[:, 0, :]
            image_embeds = torch.nn.functional.normalize(image_embeds, dim=-1)

        text_embeds = None
        if instruction is not None:
            tokenized = self.tokenizer(
                instruction,
                padding='max_length',
                max_length=100,
                truncation=True,
                return_tensors="pt"
            ).to(device)
            text_outputs = self.text_encoder(**tokenized)
            text_embeds = text_outputs.last_hidden_state[:, 0, :]
            text_embeds = torch.nn.functional.normalize(text_embeds, dim=-1)

        return image_embeds, text_embeds

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path, map_location='cpu', weights_only=True)
        # 检查是否嵌套了 'state_dict'
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        
        loaded_layers = 0
        total_layers = len(param_dict)

        for key, value in param_dict.items():
            key = key.replace('module.', '')  # 移除 DataParallel 的前缀
            if key in self.state_dict() and self.state_dict()[key].shape == value.shape:
                self.state_dict()[key].copy_(value)
                loaded_layers += 1
            else:
                print(f"Skipping {key}: not found in model or shape mismatch")

        print(f'Loaded {loaded_layers} / {total_layers} layers from {trained_path}')