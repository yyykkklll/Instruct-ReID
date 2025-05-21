import logging
from pathlib import Path
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, ViTModel
from ..utils.serialization import copy_state_dict
from .fusion import get_fusion_module

# 设置transformers库的日志级别为ERROR，减少不必要的日志输出
logging.getLogger("transformers").setLevel(logging.ERROR)

class DisentangleModule(nn.Module):
    def __init__(self, dim):
        """
        特征分离模块，将输入特征分解为身份特征和服装特征。

        Args:
            dim (int): 输入特征的维度。
        """
        super().__init__()
        self.id_linear = nn.Linear(dim, dim)  # 身份特征线性变换
        self.id_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=4)  # 身份特征自注意力
        self.id_norm = nn.LayerNorm(dim)  # 身份特征归一化
        self.cloth_linear = nn.Linear(dim, dim)  # 服装特征线性变换
        self.cloth_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=4)  # 服装特征自注意力
        self.cloth_norm = nn.LayerNorm(dim)  # 服装特征归一化
        self.id_cross_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=4)  # 身份-服装交叉注意力
        self.cloth_cross_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=4)  # 服装-身份交叉注意力
        self.id_cross_norm = nn.LayerNorm(dim)  # 身份交叉特征归一化
        self.cloth_cross_norm = nn.LayerNorm(dim)  # 服装交叉特征归一化
        self.gate = nn.Sequential(
            nn.Linear(dim*2, dim),  # 门控机制输入维度为2*dim
            nn.Sigmoid()  # 输出门控权重
        )

    def forward(self, x):
        """
        前向传播，将输入特征分解为身份特征和服装特征，并通过门控机制加权。

        Args:
            x (torch.Tensor): 输入特征，形状为 [batch_size, dim]。

        Returns:
            tuple: (id_feat, cloth_feat, gate)，身份特征、服装特征和门控权重。
        """
        batch_size, dim = x.size()
        id_feat = self.id_linear(x)
        id_feat = id_feat.unsqueeze(0)  # [1, batch_size, dim]
        id_attn, _ = self.id_attn(query=id_feat, key=id_feat, value=id_feat)
        id_feat = id_attn.squeeze(0)  # [batch_size, dim]
        id_feat = id_feat + x  # 残差连接
        id_feat = self.id_norm(id_feat)
        cloth_feat = self.cloth_linear(x)
        cloth_feat = cloth_feat.unsqueeze(0)
        cloth_attn, _ = self.cloth_attn(query=cloth_feat, key=cloth_feat, value=cloth_feat)
        cloth_feat = cloth_attn.squeeze(0)
        cloth_feat = cloth_feat + x
        cloth_feat = self.cloth_norm(cloth_feat)
        id_cross, _ = self.id_cross_attn(query=id_feat.unsqueeze(0), key=cloth_feat.unsqueeze(0), value=cloth_feat.unsqueeze(0))
        cloth_cross, _ = self.cloth_cross_attn(query=cloth_feat.unsqueeze(0), key=id_feat.unsqueeze(0), value=id_feat.unsqueeze(0))
        id_cross = id_cross.squeeze(0)
        cloth_cross = cloth_cross.squeeze(0)
        id_feat = self.id_cross_norm(id_cross + id_feat)
        cloth_feat = self.cloth_cross_norm(cloth_cross + cloth_feat)
        gate = self.gate(torch.cat([id_feat, cloth_feat], dim=-1))
        id_feat = gate * id_feat
        cloth_feat = (1 - gate) * cloth_feat
        return id_feat, cloth_feat, gate

class T2IReIDModel(nn.Module):
    def __init__(self, net_config):
        """
        文本-图像行人重识别模型，结合BERT和ViT进行多模态特征提取与融合。

        Args:
            net_config (dict): 模型配置字典，包含BERT路径、ViT路径、融合模块配置等。
        """
        super().__init__()
        self.net_config = net_config
        bert_base_path = Path(net_config.get('bert_base_path', 'pretrained/bert-base-uncased'))
        vit_base_path = Path(net_config.get('vit_pretrained', 'pretrained/vit-base-patch16-224'))
        fusion_config = net_config.get('fusion', {})
        num_classes = net_config.get('num_classes', 8000)

        # 验证预训练模型路径
        if not bert_base_path.exists() or not vit_base_path.exists():
            raise FileNotFoundError(f"Model path not found: {bert_base_path} or {vit_base_path}")

        # 初始化文本编码器和分词器
        self.tokenizer = BertTokenizer.from_pretrained(str(bert_base_path), do_lower_case=True, use_fast=True)
        self.text_encoder = BertModel.from_pretrained(str(bert_base_path))
        self.text_width = self.text_encoder.config.hidden_size  # BERT隐藏层维度（768）

        # 初始化图像编码器
        self.visual_encoder = ViTModel.from_pretrained(str(vit_base_path))

        # 初始化特征分离模块
        self.disentangle = DisentangleModule(dim=self.text_width)

        # 初始化身份分类器
        self.id_classifier = nn.Linear(self.text_width, num_classes)

        # 初始化共享MLP
        self.shared_mlp = nn.Linear(self.text_width, 512)

        # 初始化图像特征MLP
        self.image_mlp = nn.Sequential(
            nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 256)
        )

        # 初始化文本特征MLP
        self.text_mlp = nn.Sequential(
            nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 256)
        )

        # 新增文本特征自注意力模块
        self.text_attn = nn.MultiheadAttention(embed_dim=self.text_width, num_heads=4, dropout=0.1)
        self.text_attn_norm = nn.LayerNorm(self.text_width)

        # 初始化融合模块
        self.fusion = get_fusion_module(fusion_config) if fusion_config else None
        self.feat_dim = fusion_config.get("output_dim", 256) if fusion_config else 256

        # 初始化可学习的缩放参数
        self.scale = nn.Parameter(torch.ones(1), requires_grad=True)

        # 日志记录初始化信息
        logging.info(f"Initialized model with scale: {self.scale.item():.4f}, fusion: {fusion_config.get('type', 'None')}")

        # 文本分词结果缓存
        self.text_cache = {}

    def encode_image(self, image):
        """
        编码图像，提取图像特征并进行标准化。

        Args:
            image (torch.Tensor): 输入图像，形状为 [batch_size, channels, height, width] 或更高维。

        Returns:
            torch.Tensor: 标准化后的图像嵌入，形状为 [batch_size, 256]。
        """
        if image is None:
            return None
        device = next(self.parameters()).device
        if image.dim() == 5:
            image = image.squeeze(-1)
        image = image.to(device)
        image_outputs = self.visual_encoder(image)
        image_embeds = image_outputs.last_hidden_state[:, 0, :]  # 提取CLS token
        id_embeds, _, _ = self.disentangle(image_embeds)
        image_embeds = self.shared_mlp(id_embeds)
        image_embeds = self.image_mlp(image_embeds)
        image_embeds = torch.nn.functional.normalize(image_embeds, dim=-1)
        return image_embeds

    def encode_text(self, instruction):
        """
        编码文本，提取文本特征并进行标准化，新增多头自注意力模块增强特征交互。

        Args:
            instruction (str or list): 输入文本，单个字符串或字符串列表。

        Returns:
            torch.Tensor: 标准化后的文本嵌入，形状为 [batch_size, 256] 或 [256]（单文本）。
        """
        if instruction is None:
            return None
        device = next(self.parameters()).device
        if isinstance(instruction, list):
            texts = instruction
        else:
            texts = [instruction]

        # 检查缓存以复用分词结果
        cache_key = tuple(texts)
        if cache_key in self.text_cache:
            tokenized = self.text_cache[cache_key]
        else:
            tokenized = self.tokenizer(
                texts,
                padding='max_length',
                max_length=64,  # 适合CUHK-PEDES数据集的文本长度
                truncation=True,
                return_tensors="pt",
                return_attention_mask=True
            )
            self.text_cache[cache_key] = tokenized

        input_ids = tokenized['input_ids'].to(device)
        attention_mask = tokenized['attention_mask'].to(device)

        # BERT编码，禁用梯度以提升效率
        with torch.no_grad():
            text_outputs = self.text_encoder(input_ids, attention_mask=attention_mask)
        text_embeds = text_outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]

        # 多头自注意力处理，增强序列特征交互
        text_embeds = text_embeds.transpose(0, 1)  # [seq_len, batch_size, hidden_size]
        text_attn, _ = self.text_attn(
            query=text_embeds,
            key=text_embeds,
            value=text_embeds,
            key_padding_mask=~attention_mask.bool()  # 转换为布尔掩码，忽略填充token
        )
        text_embeds = text_attn.transpose(0, 1) + text_embeds.transpose(0, 1)  # 残差连接
        text_embeds = self.text_attn_norm(text_embeds)  # 归一化

        # 均值池化，结合attention_mask忽略填充token
        attention_mask = attention_mask.unsqueeze(-1)  # [batch_size, seq_len, 1]
        text_embeds = torch.sum(text_embeds * attention_mask, dim=1) / torch.sum(attention_mask, dim=1)
        # 形状: [batch_size, hidden_size]

        # 降维和标准化
        text_embeds = self.shared_mlp(text_embeds)
        text_embeds = self.text_mlp(text_embeds)
        text_embeds = torch.nn.functional.normalize(text_embeds, dim=-1)

        if not isinstance(instruction, list):
            text_embeds = text_embeds.squeeze(0)
        return text_embeds

    def forward(self, image=None, cloth_instruction=None, id_instruction=None):
        """
        前向传播，处理图像和文本输入，输出多模态特征和分类结果。

        Args:
            image (torch.Tensor, optional): 输入图像，形状为 [batch_size, channels, height, width]。
            cloth_instruction (str or list, optional): 服装描述文本。
            id_instruction (str or list, optional): 身份描述文本。

        Returns:
            tuple: (image_embeds, id_text_embeds, fused_embeds, id_logits, id_embeds,
                    cloth_embeds, cloth_text_embeds, cloth_image_embeds, gate, gate_weights)
        """
        device = next(self.parameters()).device
        id_logits, id_embeds, cloth_embeds, gate = None, None, None, None
        if image is not None:
            if image.dim() == 5:
                image = image.squeeze(-1)
            image = image.to(device)
            image_outputs = self.visual_encoder(image)
            image_embeds = image_outputs.last_hidden_state[:, 0, :]
            id_embeds, cloth_embeds, gate = self.disentangle(image_embeds)
            id_logits = self.id_classifier(id_embeds)
            image_embeds = self.shared_mlp(id_embeds)
            image_embeds = self.image_mlp(image_embeds)
            image_embeds = torch.nn.functional.normalize(image_embeds, dim=-1)
            cloth_image_embeds = self.shared_mlp(cloth_embeds)
            cloth_image_embeds = self.image_mlp(cloth_image_embeds)
            cloth_image_embeds = torch.nn.functional.normalize(cloth_image_embeds, dim=-1)
        else:
            image_embeds = None
            cloth_image_embeds = None
        cloth_text_embeds = self.encode_text(cloth_instruction)
        id_text_embeds = self.encode_text(id_instruction)
        fused_embeds, gate_weights = None, None
        if self.fusion and image_embeds is not None and id_text_embeds is not None:
            fused_embeds, gate_weights = self.fusion(image_embeds, id_text_embeds)
            fused_embeds = self.scale * torch.nn.functional.normalize(fused_embeds, dim=-1)
        return (image_embeds, id_text_embeds, fused_embeds, id_logits, id_embeds,
                cloth_embeds, cloth_text_embeds, cloth_image_embeds, gate, gate_weights)

    def load_param(self, trained_path):
        """
        加载预训练模型参数。

        Args:
            trained_path (str): 预训练模型文件路径。

        Returns:
            T2IReIDModel: 加载参数后的模型。
        """
        trained_path = Path(trained_path)
        checkpoint = torch.load(trained_path, map_location='cpu', weights_only=True)
        state_dict = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))
        self = copy_state_dict(state_dict, self)
        logging.info(f"Loaded checkpoint from {trained_path}, scale: {self.scale.item():.4f}")
        return self
