import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossModalAlign(nn.Module):
    def __init__(self, feat_dim, hidden_dim=512):
        super().__init__()
        # 图像特征投影器
        self.image_projector = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, feat_dim)
        )
        # 文本特征投影器
        self.text_projector = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, feat_dim)
        )
        # 模态判别器
        self.domain_classifier = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 2)  # 输出2类：图像或文本
        )

    def forward(self, img_feat, txt_feat):
        # 投影特征
        proj_img = self.image_projector(img_feat)  # [batch_size, feat_dim]
        proj_txt = self.text_projector(txt_feat)   # [batch_size, feat_dim]
        
        # 模态判别器输入
        combined_feat = torch.cat([proj_img, proj_txt], dim=0)  # [2*batch_size, feat_dim]
        domain_pred = self.domain_classifier(combined_feat)     # [2*batch_size, 2]
        
        return proj_img, proj_txt, domain_pred