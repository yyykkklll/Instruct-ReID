import torch
import torch.nn as nn

<<<<<<< HEAD
class AttentionGatedFusion(nn.Module):
    """结合跨模态注意力和门控机制的融合模块，用于整合图像和文本特征。"""

    def __init__(self, dim, num_heads=4, output_dim=256, dropout=0.1):
        """初始化融合模块。

        Args:
            dim: 输入特征维度。
            num_heads: 注意力头数。
            output_dim: 输出特征维度。
            dropout: Dropout 比率。
        """
        super().__init__()
        self.image_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout)
        self.text_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout)
        self.norm_image = nn.LayerNorm(dim)
        self.norm_text = nn.LayerNorm(dim)
=======

class AttentionGatedFusion(nn.Module):
    """
    结合跨模态注意力和门控机制的融合模块，用于整合图像和文本特征
    """

    def __init__(self, dim=768, num_heads=4, output_dim=512, dropout=0.1):
        """
        初始化融合模块

        Args:
            dim (int): 输入特征维度
            num_heads (int): 注意力头数
            output_dim (int): 输出特征维度
            dropout (float): Dropout 比率
        """
        super().__init__()
        # 跨模态注意力：图像引导和文本引导
        self.image_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout)
        self.text_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout)

        # 层归一化
        self.norm_image = nn.LayerNorm(dim)
        self.norm_text = nn.LayerNorm(dim)

        # 门控网络
>>>>>>> ae1d583f71d5b97df29d9414fb60417d2714e12b
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
<<<<<<< HEAD
            nn.Linear(dim, 2),
            nn.Softmax(dim=-1)
        )
=======
            nn.Linear(dim, 2),  # 输出两个权重（图像和文本）
            nn.Softmax(dim=-1)
        )

        # 最终投影层
>>>>>>> ae1d583f71d5b97df29d9414fb60417d2714e12b
        self.fc = nn.Linear(dim, output_dim)
        self.norm_final = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, image_features, text_features):
<<<<<<< HEAD
        """前向传播，生成融合特征。

        Args:
            image_features: 图像特征，形状为 [batch_size, dim]。
            text_features: 文本特征，形状为 [batch_size, dim]。

        Returns:
            fused_features: 融合特征，形状为 [batch_size, output_dim]。
        """
        image_query = image_features.unsqueeze(0)
        text_kv = text_features.unsqueeze(0)
        image_attn_output, _ = self.image_attn(image_query, text_kv, text_kv)
        image_attn_output = self.norm_image(image_attn_output.squeeze(0) + image_features)

=======
        """
        前向传播，生成融合特征

        Args:
            image_features: 图像特征，形状为 [batch_size, dim]
            text_features: 文本特征，形状为 [batch_size, dim]

        Returns:
            fused_features: 融合特征，形状为 [batch_size, output_dim]
        """
        # 跨模态注意力
        # 图像引导：以图像为query，文本为key/value
        image_query = image_features.unsqueeze(0)  # [1, batch_size, dim]
        text_kv = text_features.unsqueeze(0)  # [1, batch_size, dim]
        image_attn_output, _ = self.image_attn(image_query, text_kv, text_kv)
        image_attn_output = self.norm_image(image_attn_output.squeeze(0) + image_features)

        # 文本引导：以文本为query，图像为key/value
>>>>>>> ae1d583f71d5b97df29d9414fb60417d2714e12b
        text_query = text_features.unsqueeze(0)
        image_kv = image_features.unsqueeze(0)
        text_attn_output, _ = self.text_attn(text_query, image_kv, image_kv)
        text_attn_output = self.norm_text(text_attn_output.squeeze(0) + text_features)

<<<<<<< HEAD
        concat_features = torch.cat([image_attn_output, text_attn_output], dim=-1)
        gate_weights = self.gate(concat_features)
        image_weight, text_weight = gate_weights[:, 0:1], gate_weights[:, 1:2]

        fused_features = image_weight * image_attn_output + text_weight * text_attn_output
        fused_features = self.dropout(fused_features)
=======
        # 门控机制
        concat_features = torch.cat([image_attn_output, text_attn_output], dim=-1)  # [batch_size, dim*2]
        gate_weights = self.gate(concat_features)  # [batch_size, 2]
        image_weight, text_weight = gate_weights[:, 0:1], gate_weights[:, 1:2]  # [batch_size, 1]

        # 加权融合
        fused_features = image_weight * image_attn_output + text_weight * text_attn_output
        fused_features = self.dropout(fused_features)

        # 投影到目标维度
>>>>>>> ae1d583f71d5b97df29d9414fb60417d2714e12b
        fused_features = self.fc(fused_features)
        fused_features = self.norm_final(fused_features)

        return fused_features

<<<<<<< HEAD
def get_fusion_module(config):
    """动态创建融合模块。

    Args:
        config: 融合模块配置字典，包含 'type'、'dim' 等字段。

    Returns:
        nn.Module: 融合模块实例。

    Raises:
        ValueError: 如果 fusion_type 未知。
    """
    fusion_type = config.get("type")
    if fusion_type == "attention_gated":
        valid_params = {k: v for k, v in config.items() if k in ['dim', 'num_heads', 'output_dim', 'dropout']}
        return AttentionGatedFusion(**valid_params)
    else:
        raise ValueError(f"Unknown fusion type: {fusion_type}")
=======

def get_fusion_module(config):
    """
    动态创建融合模块

    Args:
        config: 融合模块配置字典，包含 'type' 等字段

    Returns:
        nn.Module: 融合模块实例

    Raises:
        ValueError: 如果 fusion_type 未知
    """
    fusion_type = config.get("type")
    if fusion_type == "attention_gated":
        # 过滤掉 'type' 字段，仅传递支持的参数
        valid_params = {k: v for k, v in config.items() if k in ['dim', 'num_heads', 'output_dim', 'dropout']}
        return AttentionGatedFusion(**valid_params)
    else:
        raise ValueError(f"Unknown fusion type: {fusion_type}")
>>>>>>> ae1d583f71d5b97df29d9414fb60417d2714e12b
