from __future__ import absolute_import

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, ViTModel


class T2IReIDModel(nn.Module):
    def __init__(self, net_config):
        super(T2IReIDModel, self).__init__()
        self.net_config = net_config

        # 文本编码器：BERT
        bert_base_path = net_config.bert_base_path
        self.tokenizer = BertTokenizer.from_pretrained(bert_base_path)
        self.text_encoder = BertModel.from_pretrained(bert_base_path)
        self.text_width = self.text_encoder.config.hidden_size  # 768

        # 图像编码器：ViT
        self.visual_encoder = ViTModel.from_pretrained("reid/vit-base-patch16-224")
        self.feat_dim = 768  # ViT 和 BERT 的嵌入维度

    def forward(self, image=None, instruction=None):
        device = next(self.parameters()).device

        # 图像特征
        image_embeds = None
        if image is not None:
            if image.dim() == 5:
                image = image.squeeze(-1)  # 调整形状
            image_outputs = self.visual_encoder(image)
            image_embeds = image_outputs.last_hidden_state[:, 0, :]  # [batch, 768], [CLS] token
            image_embeds = torch.nn.functional.normalize(image_embeds, dim=-1)  # 归一化

        # 文本特征
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
            text_embeds = text_outputs.last_hidden_state[:, 0, :]  # [batch, 768], [CLS] token
            text_embeds = torch.nn.functional.normalize(text_embeds, dim=-1)  # 归一化

        return image_embeds, text_embeds

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
