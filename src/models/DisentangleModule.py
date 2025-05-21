import torch
import torch.nn as nn

class DisentangleModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # 身份特征通道
        self.id_branch = nn.Sequential(
            nn.Linear(dim, dim),
            nn.MultiheadAttention(embed_dim=dim, num_heads=4),
            nn.LayerNorm(dim)
        )
        # 衣物特征通道
        self.cloth_branch = nn.Sequential(
            nn.Linear(dim, dim),
            nn.MultiheadAttention(embed_dim=dim, num_heads=4),
            nn.LayerNorm(dim)
        )
        # 交叉注意力：身份查询，衣物键/值
        self.id_cross_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=4)
        self.cloth_cross_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=4)
        # 额外的层归一化用于交叉注意力输出
        self.id_cross_norm = nn.LayerNorm(dim)
        self.cloth_cross_norm = nn.LayerNorm(dim)
        # 特征竞争机制
        self.gate = nn.Sequential(
            nn.Linear(dim*2, dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 身份分支：自注意力 + 残差连接
        id_feat = self.id_branch(x)
        id_feat = id_feat + x  # 残差连接
        id_feat = nn.LayerNorm(x.size(-1)).to(x.device)(id_feat)  # 额外归一化

        # 衣物分支：自注意力 + 残差连接
        cloth_feat = self.cloth_branch(x)
        cloth_feat = cloth_feat + x  # 残差连接
        cloth_feat = nn.LayerNorm(x.size(-1)).to(x.device)(cloth_feat)  # 额外归一化

        # 交叉注意力：分支间交互
        # 身份分支使用衣物特征作为键/值
        id_feat_attn, _ = self.id_cross_attn( # Renamed to avoid overwriting id_feat prematurely
            query=id_feat.unsqueeze(0),  # [1, batch_size, dim]
            key=cloth_feat.unsqueeze(0),
            value=cloth_feat.unsqueeze(0) #  <-- 修复在此
        )
        # 衣物分支使用身份特征作为键/值
        cloth_feat_attn, _ = self.cloth_cross_attn( # Renamed to avoid overwriting cloth_feat prematurely
            query=cloth_feat.unsqueeze(0),
            key=id_feat.unsqueeze(0),
            value=id_feat.unsqueeze(0)
        )
        id_feat_attn = id_feat_attn.squeeze(0)  # [batch_size, dim]
        cloth_feat_attn = cloth_feat_attn.squeeze(0)

        # 残差连接和层归一化
        id_feat = self.id_cross_norm(id_feat + id_feat_attn)
        cloth_feat = self.cloth_cross_norm(cloth_feat + cloth_feat_attn)

        # 动态门控竞争
        gate = self.gate(torch.cat([id_feat, cloth_feat], dim=-1))
        id_feat_gated = gate * id_feat
        cloth_feat_gated = (1 - gate) * cloth_feat

        return id_feat_gated, cloth_feat_gated, gate # 返回门控向量用于正则化损失
