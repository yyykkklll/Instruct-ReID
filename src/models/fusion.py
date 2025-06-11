import torch
import torch.nn as nn
from mamba_ssm import Mamba

class EnhancedMambaFusion(nn.Module):
    """优化后的 Mamba SSM 融合模块，门控机制前置，用于高效整合图像和文本特征。"""
    
    def __init__(self, dim, d_state=16, d_conv=4, num_layers=3, output_dim=256, dropout=0.1):
        """
        初始化增强 Mamba 融合模块。

        Args:
            dim (int): 输入特征维度（图像和文本特征的维度）。
            d_state (int): Mamba SSM 的状态维度。
            d_conv (int): Mamba SSM 的卷积核大小。
            num_layers (int): Mamba 层数，默认为 3。
            output_dim (int): 输出特征维度，默认为 256。
            dropout (float): Dropout 比率，默认为 0.1。
        """
        super().__init__()
        # 模态对齐层：三层 MLP，带 ReLU 和 LayerNorm
        self.image_align = nn.Sequential(
            nn.Linear(dim, dim),  # 第一层线性变换
            nn.ReLU(),
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),  # 第二层线性变换
            nn.ReLU(),
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),  # 第三层线性变换
            nn.ReLU(),
            nn.LayerNorm(dim)
        )
        self.text_align = nn.Sequential(
            nn.Linear(dim, dim),  # 第一层线性变换
            nn.ReLU(),
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),  # 第二层线性变换
            nn.ReLU(),
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),  # 第三层线性变换
            nn.ReLU(),
            nn.LayerNorm(dim)
        )
        
        # 前置门控机制：基于对齐特征生成权重
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),  # 输入拼接后的图像和文本特征
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim // 2),  # 降维增强非线性
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim // 2, 2),  # 输出图像和文本权重
            nn.Softmax(dim=-1)
        )
        
        # 多层 Mamba SSM：三层 Mamba 处理加权拼接特征
        self.mamba_layers = nn.ModuleList([
            Mamba(
                d_model=dim * 2,
                d_state=d_state,
                d_conv=d_conv,
                expand=2
            ) for _ in range(num_layers)
        ])
        self.mamba_norms = nn.ModuleList([nn.LayerNorm(dim * 2) for _ in range(num_layers)])
        
        # 输出投影：将 Mamba 输出投影到目标维度
        self.fc = nn.Linear(dim * 2, output_dim)
        self.norm_final = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, image_features, text_features):
        """
        前向传播，生成融合特征。

        Args:
            image_features (torch.Tensor): 图像特征，形状为 [batch_size, dim]。
            text_features (torch.Tensor): 文本特征，形状为 [batch_size, dim]。

        Returns:
            tuple: (fused_features, gate_weights)
                - fused_features: 融合特征，形状为 [batch_size, output_dim]。
                - gate_weights: 门控权重，形状为 [batch_size, 2]，用于正则化损失。
        """
        # 模态对齐：通过三层 MLP 处理图像和文本特征
        image_features = self.image_align(image_features)  # [batch_size, dim]
        text_features = self.text_align(text_features)    # [batch_size, dim]
        
        # 前置门控：基于对齐特征生成权重
        concat_features = torch.cat([image_features, text_features], dim=-1)  # [batch_size, dim*2]
        gate_weights = self.gate(concat_features)  # [batch_size, 2]
        image_weight, text_weight = gate_weights[:, 0:1], gate_weights[:, 1:2]  # [batch_size, 1]
        
        # 加权拼接：对对齐特征进行加权后拼接
        weighted_image = image_weight * image_features  # [batch_size, dim]
        weighted_text = text_weight * text_features    # [batch_size, dim]
        weighted_features = torch.cat([weighted_image, weighted_text], dim=-1)  # [batch_size, dim*2]
        weighted_features = weighted_features.unsqueeze(1)  # [batch_size, 1, dim*2]
        
        # 多层 Mamba 处理：三层 Mamba 融合加权特征
        mamba_output = weighted_features
        for mamba, norm in zip(self.mamba_layers, self.mamba_norms):
            residual = mamba_output
            mamba_output = mamba(mamba_output)  # [batch_size, 1, dim*2]
            mamba_output = norm(mamba_output + residual)  # 残差连接
        
        mamba_output = mamba_output.squeeze(1)  # [batch_size, dim*2]
        
        # 输出投影：生成最终融合特征
        fused_features = self.fc(mamba_output)  # [batch_size, output_dim]
        fused_features = self.dropout(fused_features)
        fused_features = self.norm_final(fused_features)
        
        return fused_features, gate_weights

def get_fusion_module(config):
    """
    动态创建融合模块。

    Args:
        config (dict): 融合模块配置字典，包含 'type'、'dim' 等字段。

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