import logging
from pathlib import Path
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, ViTModel
from ..utils.serialization import copy_state_dict
from .fusion import get_fusion_module

logging.getLogger("transformers").setLevel(logging.ERROR)


class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


class T2IReIDModel(nn.Module):
    def __init__(self, net_config):
        super().__init__()
        self.net_config = net_config
        self.alpha = net_config.get('grl_alpha', 1.0)  # GRL 缩放系数

        # 获取预训练模型路径
        bert_base_path = Path(net_config.get('bert_base_path', 'pretrained/bert-base-uncased'))
        vit_base_path = Path(net_config.get('vit_pretrained', 'pretrained/vit-base-patch16-224'))
        fusion_config = net_config.get('fusion', {})
        num_classes = net_config.get('num_classes', 8000)

        if not bert_base_path.exists() or not vit_base_path.exists():
            raise FileNotFoundError(f"Model path not found: {bert_base_path} or {vit_base_path}")

        # 初始化编码器
        self.tokenizer = BertTokenizer.from_pretrained(str(bert_base_path))
        self.text_encoder = BertModel.from_pretrained(str(bert_base_path))
        self.text_width = self.text_encoder.config.hidden_size
        self.visual_encoder = ViTModel.from_pretrained(str(vit_base_path))

        # 解纠缠模块
        self.id_projection = nn.Linear(self.text_width, self.text_width)
        self.id_classifier = nn.Linear(self.text_width, num_classes)
        self.cloth_projection = nn.Linear(self.text_width, self.text_width)  # 衣物投影头

        # 共享 MLP 和模态特定 MLP
        self.shared_mlp = nn.Linear(self.text_width, 512)
        self.image_mlp = nn.Sequential(
            nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 256)
        )
        self.text_mlp = nn.Sequential(
            nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 256)
        )

        # 融合模块
        self.fusion = get_fusion_module(fusion_config) if fusion_config else None
        self.feat_dim = fusion_config.get("output_dim", 256) if fusion_config else 256
        self.scale = nn.Parameter(torch.ones(1), requires_grad=True)
        logging.info(
            f"Initialized model with scale: {self.scale.item():.4f}, fusion: {fusion_config.get('type', 'None')}")

    def encode_image(self, image):
        if image is None:
            return None
        device = next(self.parameters()).device
        if image.dim() == 5:
            image = image.squeeze(-1)
        image = image.to(device)
        image_outputs = self.visual_encoder(image)
        image_embeds = image_outputs.last_hidden_state[:, 0, :]
        id_embeds = torch.nn.functional.relu(self.id_projection(image_embeds))
        image_embeds = self.shared_mlp(id_embeds)
        image_embeds = self.image_mlp(image_embeds)
        image_embeds = torch.nn.functional.normalize(image_embeds, dim=-1)
        return image_embeds

    def encode_text(self, instruction):
        if instruction is None:
            return None
        device = next(self.parameters()).device
        tokenized = self.tokenizer(
            instruction, padding='max_length', max_length=100,
            truncation=True, return_tensors="pt"
        ).to(device)
        text_outputs = self.text_encoder(**tokenized)
        text_embeds = text_outputs.last_hidden_state[:, 0, :]
        text_embeds = self.shared_mlp(text_embeds)
        text_embeds = self.text_mlp(text_embeds)
        text_embeds = torch.nn.functional.normalize(text_embeds, dim=-1)
        return text_embeds

    def forward(self, image=None, cloth_instruction=None, id_instruction=None):
        device = next(self.parameters()).device
        id_logits, id_embeds, cloth_embeds = None, None, None

        # 图像编码与解纠缠
        if image is not None:
            if image.dim() == 5:
                image = image.squeeze(-1)
            image = image.to(device)
            image_outputs = self.visual_encoder(image)
            image_embeds = image_outputs.last_hidden_state[:, 0, :]

            # 身份解纠缠
            id_embeds = torch.nn.functional.relu(self.id_projection(image_embeds))
            id_logits = self.id_classifier(id_embeds)

            # 衣物解纠缠
            cloth_embeds = self.cloth_projection(image_embeds)
            cloth_embeds = GradientReversalLayer.apply(cloth_embeds, self.alpha)
            cloth_embeds = torch.nn.functional.relu(cloth_embeds)

            # 投影到统一维度
            image_embeds = self.shared_mlp(id_embeds)
            image_embeds = self.image_mlp(image_embeds)
            image_embeds = torch.nn.functional.normalize(image_embeds, dim=-1)
        else:
            image_embeds = None

        # 文本编码（衣物和身份描述）
        cloth_text_embeds = self.encode_text(cloth_instruction)
        id_text_embeds = self.encode_text(id_instruction)

        # 融合特征（基于身份特征）
        fused_embeds = None
        if self.fusion and image_embeds is not None and id_text_embeds is not None:
            fused_embeds = self.fusion(image_embeds, id_text_embeds)
            fused_embeds = self.scale * torch.nn.functional.normalize(fused_embeds, dim=-1)

        return image_embeds, id_text_embeds, fused_embeds, id_logits, id_embeds, cloth_embeds, cloth_text_embeds

    def load_param(self, trained_path):
        trained_path = Path(trained_path)
        checkpoint = torch.load(trained_path, map_location='cpu', weights_only=True)
        state_dict = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))
        self = copy_state_dict(state_dict, self)
        logging.info(f"Loaded checkpoint from {trained_path}, scale: {self.scale.item():.4f}")
        return self