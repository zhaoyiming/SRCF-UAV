import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet3d.models import BACKBONES

@BACKBONES.register_module()
class PointWiseMLP(nn.Module):
    """
    纯逐点 MLP：将 (B, N, 6) 映射到 (B, N, 256)
    无邻域／注意力操作，速度最快
    """
    def __init__(self, in_dim=6, embed_dim=256, hidden_dims=(64, 128)):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(last, h, bias=False),
                nn.BatchNorm1d(h),
                nn.ReLU(inplace=True)
            ]
            last = h
        layers.append(nn.Linear(last, embed_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        """
        x: (B, N, 6) 6 维原始雷达属性
        -> (B, N, 256)
        """
        B, N, _ = x.shape
        x = x.reshape(B * N, -1)       # BN × 6
        x = self.mlp(x)                # BN × 256
        x = x.view(B, N, -1)           # B × N × 256
        return x


class PointMLPWithSelfAttention(nn.Module):
    """
    先 MLP → d0，再用 L 层 Self‑Attention 让点之间交互
    默认 transformer 输入格式为 (S, B, E)，需转置
    """
    def __init__(
        self,
        in_dim=6,
        embed_dim=256,
        mlp_hidden=(64, 128),
        attn_dim=128,
        num_layers=2,
        num_heads=4,
        attn_dropout=0.1
    ):
        super().__init__()
        self.mlp_head = PointWiseMLP(in_dim, attn_dim, mlp_hidden)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=attn_dim,
            nhead=num_heads,
            dropout=attn_dropout,
            batch_first=False    # PyTorch 默认 (S,B,E)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # 最终再映射到 256 维
        self.proj = nn.Linear(attn_dim, embed_dim)

    def forward(self, x):
        """
        x: (B, N, 6)
        return: (B, N, 256)
        """
        feat = self.mlp_head(x)        # (B, N, attn_dim)
        feat = feat.transpose(0, 1)    # → (N, B, attn_dim)
        feat = self.transformer(feat)  # Self‑Attention
        feat = feat.transpose(0, 1)    # → (B, N, attn_dim)
        feat = self.proj(feat)         # (B, N, 256)
        return feat
