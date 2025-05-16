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
        self.alpha = net_config.get('grl_alpha', 1.0)  # GRL 缩放因子

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
        self.text_width = self.text_encoder.config.hidden_size  # 768 for bert-base-uncased
        self.visual_encoder = ViTModel.from_pretrained(str(vit_base_path))

        # 优化1：升级投影头为3层MLP
        # 目的：增强身份和服装特征分离,提高解纠缠能力
        self.id_projection = nn.Sequential(
            nn.Linear(self.text_width, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, self.text_width),
            nn.BatchNorm1d(self.text_width)
        )
        self.cloth_projection = nn.Sequential(
            nn.Linear(self.text_width, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, self.text_width),
            nn.BatchNorm1d(self.text_width)
        )
        self.id_classifier = nn.Linear(self.text_width, num_classes)

        # 优化2：增强空间注意力模块
        # 目的：通过多头注意力和残差连接提高对服装区域的关注
        self.spatial_attention = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.text_width, self.text_width // 4),  # 降维以提高效率
                nn.ReLU(),
                nn.Linear(self.text_width // 4, 1),  # 每个patch的分数
                nn.Sigmoid()
            ) for _ in range(4)  # 4个注意力头
        ])
        self.attention_norm = nn.LayerNorm(self.text_width)  # 归一化注意力输出
        self.attention_dropout = nn.Dropout(0.1)  # 防止过拟合

        # 共享和模态特定的MLP
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

        # Patch token融合权重
        self.patch_weight = nn.Parameter(torch.tensor(0.3), requires_grad=True)
        self.cls_weight = nn.Parameter(torch.tensor(0.7), requires_grad=True)

        logging.info(
            f"Initialized model with scale: {self.scale.item():.4f}, "
            f"fusion: {fusion_config.get('type', 'None')}, "
            f"patch_weight: {self.patch_weight.item():.4f}, "
            f"cls_weight: {self.cls_weight.item():.4f}"
        )

    def update_alpha(self, epoch):
        """
        动态更新GRL alpha参数,延长预热期
        """
        self.alpha = min(1.0, 0.1 + epoch * 0.01)  # 在100个epoch内线性增加到1.0
        logging.info(f"Updated GRL alpha to {self.alpha:.4f} at epoch {epoch}")

    def encode_image(self, image):
        """
        优化的图像编码：聚合patch tokens,应用增强的空间注意力
        """
        if image is None:
            return None
        device = next(self.parameters()).device
        if image.dim() == 5:
            image = image.squeeze(-1)
        image = image.to(device)
        
        # ViT编码
        image_outputs = self.visual_encoder(image)
        image_embeds = image_outputs.last_hidden_state  # [batch_size, 197, 768]

        # 聚合patch tokens
        cls_token = image_embeds[:, 0, :]  # [batch_size, 768]
        patch_tokens = image_embeds[:, 1:, :]  # [batch_size, 196, 768]

        # 使用多头机制增强空间注意力
        attention_weights = []
        for head in self.spatial_attention:
            weights = head(patch_tokens).squeeze(-1)  # [batch_size, 196]
            weights = torch.softmax(weights, dim=-1)
            attention_weights.append(weights)
        # 平均多头权重
        attention_weights = torch.mean(torch.stack(attention_weights), dim=0)  # [batch_size, 196]
        attention_weights = self.attention_dropout(attention_weights)
        
        # 使用注意力聚合patch tokens
        patch_agg = torch.einsum('bnd,bn->bd', patch_tokens, attention_weights)  # [batch_size, 768]
        patch_agg = self.attention_norm(patch_agg)  # 归一化以提高稳定性

        # 融合CLS token和patch特征
        image_embeds = self.cls_weight * cls_token + self.patch_weight * patch_agg
        image_embeds = torch.nn.functional.normalize(image_embeds, dim=-1)

        # 使用3层MLP进行身份投影
        id_embeds = self.id_projection(image_embeds)
        
        # 投影到统一空间
        image_embeds = self.shared_mlp(id_embeds)
        image_embeds = self.image_mlp(image_embeds)
        image_embeds = torch.nn.functional.normalize(image_embeds, dim=-1)
        return image_embeds

    def encode_text(self, instruction):
        """
        文本编码：保持不变,兼容未来优化
        """
        if instruction is None:
            return None
        device = next(self.parameters()).device
        
        if isinstance(instruction, list):
            texts = instruction
        else:
            texts = [instruction]
            
        tokenized = self.tokenizer(
            texts, 
            padding='max_length', 
            max_length=100,
            truncation=True, 
            return_tensors="pt",
            return_attention_mask=True
        ).to(device)
        
        text_outputs = self.text_encoder(
            input_ids=tokenized.input_ids,
            attention_mask=tokenized.attention_mask
        )
        
        text_embeds = text_outputs.last_hidden_state[:, 0, :]
        text_embeds = self.shared_mlp(text_embeds)
        text_embeds = self.text_mlp(text_embeds)
        text_embeds = torch.nn.functional.normalize(text_embeds, dim=-1)
        
        if not isinstance(instruction, list):
            return text_embeds
        return text_embeds

    def forward(self, image=None, cloth_instruction=None, id_instruction=None):
        """
        前向传播：整合优化的图像编码,保持输出格式
        """
        device = next(self.parameters()).device
        id_logits, id_embeds, cloth_embeds = None, None, None

        if image is not None:
            if image.dim() == 5:
                image = image.squeeze(-1)
            image = image.to(device)
            image_outputs = self.visual_encoder(image)
            image_embeds = image_outputs.last_hidden_state

            # 聚合patch tokens
            cls_token = image_embeds[:, 0, :]
            patch_tokens = image_embeds[:, 1:, :]

            # 增强空间注意力
            attention_weights = []
            for head in self.spatial_attention:
                weights = head(patch_tokens).squeeze(-1)
                weights = torch.softmax(weights, dim=-1)
                attention_weights.append(weights)
            attention_weights = torch.mean(torch.stack(attention_weights), dim=0)
            attention_weights = self.attention_dropout(attention_weights)
            
            patch_agg = torch.einsum('bnd,bn->bd', patch_tokens, attention_weights)
            patch_agg = self.attention_norm(patch_agg)

            # 融合CLS token和patch特征
            image_embeds = self.cls_weight * cls_token + self.patch_weight * patch_agg
            image_embeds = torch.nn.functional.normalize(image_embeds, dim=-1)

            # 使用3层MLP投影头
            id_embeds = self.id_projection(image_embeds)
            id_logits = self.id_classifier(id_embeds)

            cloth_embeds = self.cloth_projection(image_embeds)
            cloth_embeds = GradientReversalLayer.apply(cloth_embeds, self.alpha)
            cloth_embeds = torch.nn.functional.relu(cloth_embeds)

            # 投影到统一维度
            image_embeds = self.shared_mlp(id_embeds)
            image_embeds = self.image_mlp(image_embeds)
            image_embeds = torch.nn.functional.normalize(image_embeds, dim=-1)

            cloth_image_embeds = self.shared_mlp(cloth_embeds)
            cloth_image_embeds = self.image_mlp(cloth_image_embeds)
            cloth_image_embeds = torch.nn.functional.normalize(cloth_image_embeds, dim=-1)
        else:
            image_embeds = None
            cloth_image_embeds = None

        # 文本编码
        cloth_text_embeds = self.encode_text(cloth_instruction)
        id_text_embeds = self.encode_text(id_instruction)

        # 特征融合
        fused_embeds = None
        if self.fusion and image_embeds is not None and id_text_embeds is not None:
            fused_embeds = self.fusion(image_embeds, id_text_embeds)
            fused_embeds = self.scale * torch.nn.functional.normalize(fused_embeds, dim=-1)

        return (image_embeds, id_text_embeds, fused_embeds, id_logits, id_embeds,
                cloth_embeds, cloth_text_embeds, cloth_image_embeds)

    def load_param(self, trained_path):
        """
        加载检查点：确保跨平台兼容性
        """
        trained_path = Path(trained_path)
        checkpoint = torch.load(trained_path, map_location='cpu', weights_only=True)
        state_dict = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))
        self = copy_state_dict(state_dict, self)
        logging.info(f"Loaded checkpoint from {trained_path}, scale: {self.scale.item():.4f}")
        return self