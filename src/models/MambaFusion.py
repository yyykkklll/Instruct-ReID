import torch
import torch.nn as nn
from mamba_ssm import Mamba

class EnhancedMambaFusion(nn.Module):
    """优化后的 Mamba SSM 融合模块，用于高效整合图像和文本特征。"""
    
    def __init__(self, dim, d_state=16, d_conv=4, num_layers=2, output_dim=256, dropout=0.1):
        """初始化增强 Mamba 融合模块。

        Args:
            dim: 输入特征维度（图像和文本特征的维度）。
            d_state: Mamba SSM 的状态维度。
            d_conv: Mamba SSM 的卷积核大小。
            num_layers: Mamba 层数。
            output_dim: 输出特征维度。
            dropout: Dropout 比率。
        """
        super().__init__()
        # 模态对齐层：跨模态投影
        self.image_align = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.LayerNorm(dim)
        )
        self.text_align = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.LayerNorm(dim)
        )
        
        # 多层 Mamba SSM
        self.mamba_layers = nn.ModuleList([
            Mamba(
                d_model=dim * 2,
                d_state=d_state,
                d_conv=d_conv,
                expand=2
            ) for _ in range(num_layers)
        ])
        self.mamba_norms = nn.ModuleList([nn.LayerNorm(dim * 2) for _ in range(num_layers)])
        
        # 增强门控机制：结合 Mamba 输出和原始特征
        self.gate_attn = nn.MultiheadAttention(embed_dim=dim * 2, num_heads=4, dropout=dropout)
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, 2),
            nn.Softmax(dim=-1)
        )
        
        # 输出投影
        self.fc = nn.Linear(dim * 2, output_dim)
        self.norm_final = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, image_features, text_features):
        """前向传播，生成融合特征。

        Args:
            image_features: 图像特征，形状为 [batch_size, dim]。
            text_features: 文本特征，形状为 [batch_size, dim]。

        Returns:
            fused_features: 融合特征，形状为 [batch_size, output_dim]。
            gate_weights: 门控权重，形状为 [batch_size, 2]，用于正则化损失。
        """
        # 模态对齐
        image_features = self.image_align(image_features)
        text_features = self.text_align(text_features)
        
        # 拼接特征
        concat_features = torch.cat([image_features, text_features], dim=-1)  # [batch_size, dim*2]
        concat_features = concat_features.unsqueeze(1)  # [batch_size, 1, dim*2]
        
        # 多层 Mamba 处理
        mamba_output = concat_features
        for mamba, norm in zip(self.mamba_layers, self.mamba_norms):
            residual = mamba_output
            mamba_output = mamba(mamba_output)  # [batch_size, 1, dim*2]
            mamba_output = norm(mamba_output + residual)  # 残差连接
        
        mamba_output = mamba_output.squeeze(1)  # [batch_size, dim*2]
        
        # 增强门控：结合 Mamba 输出和原始特征
        gate_input = mamba_output.unsqueeze(0)  # [1, batch_size, dim*2]
        gate_attn_output, _ = self.gate_attn(gate_input, gate_input, gate_input)
        gate_attn_output = gate_attn_output.squeeze(0)  # [batch_size, dim*2]
        gate_weights = self.gate(gate_attn_output)  # [batch_size, 2]
        image_weight, text_weight = gate_weights[:, 0:1], gate_weights[:, 1:2]
        
        # 分离 Mamba 输出
        image_part = mamba_output[:, :image_features.size(-1)]  # [batch_size, dim]
        text_part = mamba_output[:, image_features.size(-1):]   # [batch_size, dim]
        
        # 加权融合
        fused_features = image_weight * image_part + text_weight * text_part
        fused_features = self.dropout(fused_features)
        
        # 输出投影
        fused_features = self.fc(mamba_output)  # 使用完整 Mamba 输出
        fused_features = self.norm_final(fused_features)
        
        return fused_features, gate_weights

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
    if fusion_type == "enhanced_mamba":
        valid_params = {k: v for k, v in config.items() if k in ['dim', 'd_state', 'd_conv', 'num_layers', 'output_dim', 'dropout']}
        return EnhancedMambaFusion(**valid_params)
    elif fusion_type == "mamba":
        valid_params = {k: v for k, v in config.items() if k in ['dim', 'd_state', 'd_conv', 'output_dim', 'dropout']}
        return MambaFusion(**valid_params)
    elif fusion_type == "attention_gated":
        valid_params = {k: v for k, v in config.items() if k in ['dim', 'num_heads', 'output_dim', 'dropout']}
        return AttentionGatedFusion(**valid_params)
    else:
        raise ValueError(f"Unknown fusion type: {fusion_type}")