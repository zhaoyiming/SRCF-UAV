
from __future__ import annotations
import torch
import torch.nn as nn
from typing import Optional
# from pytorch3d.ops import knn_points
__all__ = ["RangeAdaptiveRadarAggregation"]


class RangeAdaptiveRadarAggregation(nn.Module):
    """
    Range‑Adaptive Radar Aggregation (RAR) with optional speed gating.

    Args
    ----
    embed_dim     : 图像 query 特征维度
    radar_dim     : 雷达 query 特征维度
    num_heads     : 多头注意力头数
    attn_type     : "dot" | "deform"
    num_points    : deform 模式下每个 image query 选取的雷达邻居数 P
    max_distance  : xyz 距离归一化尺度
    max_speed     : 速度归一化尺度 (m/s)
    speed_weight  : λ，速度项全局权重
    dropout       : dropout 概率
    fuse          : "add" | "cat" 融合方式
    use_distance_gate : 是否启用距离/速度门控
    """

    def __init__(
        self,
        embed_dim: int = 256,
        radar_dim: int = 256,
        num_heads: int = 8,
        attn_type: str = "dot",
        num_points: int = 4,
        max_distance: float = 1.732,
        max_speed: float = 25.0,          # ★ 速度归一化尺度
        max_height: float = 1.0,         # ★ 高度归一化尺度
        speed_weight: float = 0.5,        # ★ λ
        height_weight: float = 0.1,       # ★ 高度权重
        dropout: float = 0.1,
        fuse: str = "add",
        use_distance_gate: bool = True,
        use_speed_gate: bool = True,
        distance_gate_type: str = "xyz",  # ← 新增：'xyz'(默认)=旧3D | 'xy_z'=平面+高度
        num_neighbors: int = 64,

    ) -> None:
        super().__init__()
        assert attn_type in {"dot", "deform", "dot_optimized"}, "attn_type must be 'dot' or 'deform' or 'dot_optimized'"
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        # ---------------------------- 保存超参数 ----------------------------
        self.embed_dim = embed_dim
        self.radar_dim = radar_dim
        self.num_heads = num_heads
        self.attn_type = attn_type
        self.num_points = num_points
        self.max_distance = max_distance
        self.max_speed = max_speed
        self.max_height = max_height
        self.speed_weight = speed_weight
        self.height_weight = height_weight
        self.use_distance_gate = use_distance_gate
        self.use_speed_gate = use_speed_gate
        self.distance_gate_type = distance_gate_type
        self.use_gate = self.use_distance_gate or self.use_speed_gate
    
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # --------------------------- 线性映射 ------------------------------
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(radar_dim, embed_dim)
        self.v_proj = nn.Linear(radar_dim, embed_dim)

        # ---------------------- 可学习门控平滑度 ----------------------------
        
        if use_distance_gate:
            self.sigma = nn.Parameter(torch.tensor(0.15))   # 距离项 σ
            self.sigma_xy = nn.Parameter(torch.tensor(0.15)) # 分离项 σ_sep
            self.sigma_z = nn.Parameter(torch.tensor(0.15)) # 分离项 σ_sep
        
        if use_speed_gate:
            self.sigma_v = nn.Parameter(torch.tensor(0.15)) # 速度项 σ_v

        # --------------------- 融合 & 正则化模块 ----------------------------
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)
        self.fuse = fuse
        if fuse == "cat":
            self.cat_proj = nn.Linear(embed_dim * 2, embed_dim)
        self.offset_mlp  = nn.Linear(embed_dim,   3)  # ← 新增：偏移预测
        self.num_neighbors = num_neighbors

     
    @staticmethod
    def _pos_xyz(xyz10: torch.Tensor) -> torch.Tensor:
        """提取 xyz (B,N,3)"""
        return xyz10[..., :3]

    @staticmethod
    def _l2_speed(xyz10: torch.Tensor) -> torch.Tensor:
        """提取平面速度范数 ‖(vx,vy)‖₂ → (B,N)"""
        return xyz10[..., -2:]

    def _distance_gate_sep(self,
                        dxy: torch.Tensor,
                        dz:  torch.Tensor,
                        dv:  torch.Tensor) -> torch.Tensor:
        """
        dxy : (···)  平面距离
        dz  : (···)  高度差
        dv  : (···)  速度差
        """
        gate = 1.0
        if self.use_distance_gate:
            gate *= torch.exp(
                - (dxy / self.max_distance).clamp(min=0.,max=1.) /
                torch.clamp(self.sigma_xy.abs(), min=1e-3)
            ) * torch.exp(
                - self.height_weight * (dz / self.max_height).clamp(min=0.,max=1.) /
                torch.clamp(self.sigma_z.abs(),  min=1e-3)
            )
        if self.use_speed_gate:
            gate *= torch.exp(
                - self.speed_weight *
                (dv / self.max_speed).clamp(min=0.,max=1.) /
                torch.clamp(self.sigma_v.abs(),  min=1e-3)
            )
        return gate

    def _distance_gate(self, dist: torch.Tensor, dv: torch.Tensor) -> torch.Tensor:
        """
        综合距离 + 速度门控
        dist : (···) xyz 欧氏距离
        dv   : (···) 速度差 |v_img ‑ v_rad|
        """
        if self.use_distance_gate and self.use_speed_gate:
            d_norm = torch.clamp(dist / self.max_distance, min=0.,max=1.0)
            v_norm = torch.clamp(dv   / self.max_speed, min=0.,max=1.0)

            gate = torch.exp(-d_norm / torch.clamp(self.sigma.abs(),  min=1e-3)) * \
                torch.exp(-self.speed_weight *
                            v_norm / torch.clamp(self.sigma_v.abs(), min=1e-3))
        elif self.use_distance_gate:
            d_norm = torch.clamp(dist / self.max_distance, min=0.,max=1.0)
            gate = torch.exp(-d_norm / torch.clamp(self.sigma.abs(),  min=1e-3))
        elif self.use_speed_gate:
            v_norm = torch.clamp(dv   / self.max_speed, min=0.,max=1.0)
            gate = torch.exp(-self.speed_weight *
                             v_norm / torch.clamp(self.sigma_v.abs(), min=1e-3))
        else:
            gate = torch.ones_like(dist)
        return gate  # (···)


    def _forward_dot(self, q, k, v, img_xyz10, radar_xyz10, point_mask=None):
        B, H, Q, _ = q.shape

        # 拆分位置和速度
        img_pos, rad_pos = self._pos_xyz(img_xyz10), self._pos_xyz(radar_xyz10)  # (B,?,?,3)
        img_vel, rad_vel = self._l2_speed(img_xyz10), self._l2_speed(radar_xyz10)  # (B,?)

        if self.distance_gate_type == "xyz":
            dist = torch.cdist(img_pos, rad_pos)                                     # (B,Q,R)
        elif self.distance_gate_type == "xy_z":
            dxy    = torch.cdist(img_pos[..., :2], rad_pos[..., :2])  # (B,Q,R)
            dz     = (img_pos[..., 2].unsqueeze(2) - rad_pos[..., 2].unsqueeze(1)).abs()
        else:
            raise ValueError(f"Invalid distance gate type: {self.distance_gate_type}")
        dv = torch.linalg.norm(img_vel.unsqueeze(2) - rad_vel.unsqueeze(1),dim=-1)              # (B,Q,R)

        attn_logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale          # (B,H,Q,R)
        if self.use_gate:
            if self.distance_gate_type == "xyz":
                gate = self._distance_gate(dist, dv)
            elif self.distance_gate_type == "xy_z":
                gate = self._distance_gate_sep(dxy, dz, dv)
            else:
                raise ValueError(f"Invalid distance gate type: {self.distance_gate_type}")
            attn_logits = attn_logits + torch.log(gate.unsqueeze(1) + 1e-6)
        
        if point_mask is not None:
            m = point_mask
            if m.dim() == 3 and m.size(-1) == 1:  # (B,R,1) -> (B,R)
                m = m.squeeze(-1)
            m = (m > 0).to(dtype=attn_logits.dtype)   # 0/1 浮点
            m = m.unsqueeze(1).unsqueeze(1)           # (B,1,1,R) 广播到 (B,H,Q,R)
            attn_logits = attn_logits + torch.log(m + 1e-6)

        attn = torch.softmax(attn_logits, dim=-1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, v)                                          # (B,H,Q,D_h)
        return context.transpose(1, 2).reshape(B, Q, self.embed_dim)            # (B,Q,D)
    
    def _forward_topk_dot(self, q, k, v, img_xyz10, radar_xyz10, point_mask=None):
        B, H, Q, D_h = q.shape
        _, _, R, _ = k.shape # k 的形状通常是 (B, H, R, D_h)
        K = self.num_neighbors # 需要选择的邻居数量
        # print(K)

        # 1. 拆分位置和速度
        img_pos, rad_pos = self._pos_xyz(img_xyz10), self._pos_xyz(radar_xyz10)
        img_vel, rad_vel = self._l2_speed(img_xyz10), self._l2_speed(radar_xyz10)

        # 2. 计算距离并找到 Top-K 最近邻
        # 关键步骤：基于物理距离选择K个最近邻
        dist = torch.cdist(img_pos, rad_pos)  # (B, Q, R)
        
        # 获取K个最近邻的距离和索引
        # topk_dists 将用于门控, topk_indices 将用于汇集数据
        topk_dists, topk_indices = torch.topk(dist, k=K, dim=-1, largest=False) # (B, Q, K)
        idx_for_k_v = topk_indices.unsqueeze(1).unsqueeze(-1).expand(-1, H, -1, -1, D_h)
        B_idx = torch.arange(B, device=k.device).view(B, 1, 1, 1)
        H_idx = torch.arange(H, device=k.device).view(1, H, 1, 1)
        indices_exp = topk_indices.view(B, 1, Q, K) # Shape: (B, 1, Q, K)
        k_topk = k[B_idx, H_idx, indices_exp] # Output shape: (B, H, Q, K, D_h)
        v_topk = v[B_idx, H_idx, indices_exp] # Output shape: (B, H, Q, K, D_h)
        rad_vel_topk = rad_vel[B_idx.squeeze(-1), indices_exp.squeeze(1)] # Output shape: (B, Q, K, 2)
        # 4. 计算 Top-K 点的门控 (Gating)
        # 只为被选中的K个邻居计算速度差
        dv_topk = (img_vel.unsqueeze(2) - rad_vel_topk).norm(dim=-1) # (B, Q, K)
        
        # 使用Top-K点的距离和速度差来创建门控
        gate = self._distance_gate(topk_dists, dv_topk) # (B, Q, K)
        
        # 5. 计算注意力 (在Top-K点上进行)
        # attn_logits: (B, H, Q, 1, D_h) @ (B, H, Q, D_h, K) -> (B, H, Q, K)
        attn_logits = (q.unsqueeze(3) @ k_topk.transpose(-2, -1)).squeeze(3) * self.scale

        # 将门控应用到注意力logit上
        attn_logits = attn_logits + torch.log(gate.unsqueeze(1) + 1e-6)

        # 6. 标准化并聚合
        attn = torch.softmax(attn_logits, dim=-1) # 在K维度上进行Softmax
        attn = self.dropout(attn)

        # context: (B, H, Q, 1, K) @ (B, H, Q, K, D_h) -> (B, H, Q, D_h)
        context = (attn.unsqueeze(3) @ v_topk).squeeze(3)
        
        return context.transpose(1, 2).reshape(B, Q, self.embed_dim)

    # ------------------------------------------------------------------
    # Top‑K dot‑product attention (显存友好 + 数值稳定)
    # ------------------------------------------------------------------
    def _forward_topk_dot_gather(
        self,
        q: torch.Tensor,               # (B,H,Q,D_h)
        k: torch.Tensor,               # (B,H,R,D_h)
        v: torch.Tensor,               # (B,H,R,D_h)
        img_xyz10: torch.Tensor,       # (B,Q,10)  其中 [:3] 是 xyz, [-2:] 是 vx,vy
        radar_xyz10: torch.Tensor      # (B,R,10)
    ) -> torch.Tensor:
        """
        1. 仅在 Top‑K 最近邻上做点积注意力 → 减少计算 / 显存
        2. 用 torch.gather 按 int32 索引抽取 k,v → 不复制大张量
        3. soft‑max 前减最大值，scale 降温，避免梯度爆炸
        """
        # ---------------------------------------------------------------
        B, H, Q, D_h = q.shape                # q : (B,H,Q,D_h)
        R            = k.size(2)              # k,v: (B,H,R,D_h)
        K            = min(self.num_neighbors, R)

        # 1) 位置 / 速度
        img_pos = self._pos_xyz(img_xyz10)    # (B,Q,3)
        rad_pos = self._pos_xyz(radar_xyz10)  # (B,R,3)
        img_vel = self._l2_speed(img_xyz10)   # (B,Q,2)
        rad_vel = self._l2_speed(radar_xyz10) # (B,R,2)

        # 2) 欧氏距离并取 Top‑K 最近邻
        dist, topk_idx = torch.cdist(img_pos, rad_pos).topk(K, dim=-1, largest=False) # (B,Q,K)

        # 3) gather k / v  (保持显存友好)
        k_exp   = k.unsqueeze(2).expand(-1, -1, Q, -1, -1)                        # (B,H,Q,R,D_h)
        v_exp   = v_exp = v.unsqueeze(2).expand_as(k_exp)
        idx_exp = topk_idx.unsqueeze(1).unsqueeze(-1).expand(-1, H, -1, -1, D_h)  # (B,H,Q,K,D_h)
        k_topk  = torch.gather(k_exp, 3, idx_exp)                                 # (B,H,Q,K,D_h)
        v_topk  = torch.gather(v_exp, 3, idx_exp)                                 # (B,H,Q,K,D_h)

        # 4) gather 对应的雷达速度
        # rad_vel : (B,R,2) -> 先扩展到 (B,Q,R,2)，再在 dim=2 采样
        rad_vel_exp  = rad_vel.unsqueeze(1).expand(-1, Q, -1, -1)                 # (B,Q,R,2)
        idx_exp_vel  = topk_idx.unsqueeze(-1).expand(-1, -1, -1, rad_vel.size(-1))# (B,Q,K,2)
        rad_vel_topk = torch.gather(rad_vel_exp, 2, idx_exp_vel)                  # (B,Q,K,2)

        # 5) 计算速度差并生成门控
        dv_topk = (img_vel.unsqueeze(2) - rad_vel_topk).norm(dim=-1)              # (B,Q,K)
        gate    = self._distance_gate(dist, dv_topk)                              # (B,Q,K)

        # 6) 注意力 (减最大值保证数值稳定；self.scale 已在 __init__ 乘 0.5)
        attn_logits = (q.unsqueeze(3) @ k_topk.transpose(-2, -1)).squeeze(3) * self.scale  # (B,H,Q,K)
        attn_logits = attn_logits + torch.log(gate.unsqueeze(1) + 1e-6)
        attn_logits = attn_logits - attn_logits.max(dim=-1, keepdim=True).values   # 降温&去偏
        
        attn = torch.softmax(attn_logits, dim=-1)
        attn = self.dropout(attn)

        # 7) 聚合并返回
        context = (attn.unsqueeze(3) @ v_topk).squeeze(3)                         # (B,H,Q,D_h)
        context = context.transpose(1, 2).reshape(B, Q, self.embed_dim)           # (B,Q,D)
        return context



    def _forward_deform(self, q_raw, k_raw, v_raw, img_xyz10, radar_xyz10):
        B, Q, C = q_raw.shape
        _, R, _ = radar_xyz10.shape
        P = 1  # 保持P=1
        K = self.num_neighbors
        H = self.num_heads
        D_h = self.head_dim

        # 1. 位置和速度提取 (新增了速度提取)
        img_pos, rad_pos = self._pos_xyz(img_xyz10), self._pos_xyz(radar_xyz10)     # (B,Q,3), (B,R,3)
        img_vel, rad_vel = self._l2_speed(img_xyz10), self._l2_speed(radar_xyz10) # (B,Q,1), (B,R,1)

        # 2. 预测 P 个偏移
        offsets = self.offset_mlp(q_raw).view(B, Q, P, 3)   # (B, Q, P, 3)
        sample_pos = img_pos.unsqueeze(2) + offsets        # (B, Q, P, 3)

        # 3. cdist + topk(K) 找 K 个最近邻
        sample_flat = sample_pos.view(B, Q * P, 3)          # (B, Q*P, 3)
        dists_all = torch.cdist(sample_flat, rad_pos)       # (B, Q*P, R)
        _, idx_flat = dists_all.topk(K, dim=-1, largest=False) # (B, Q*P, K)
        idx = idx_flat.view(B, Q, P, K)                     # (B, Q, P, K)

        # 4. Gather k, v, 以及用于门控的位置和速度
        batch_off = torch.arange(B, device=q_raw.device).view(B,1,1,1) * R
        idx_off   = (idx + batch_off).view(-1)             # (B*Q*P*K)
        
        # Gather k, v
        k_flat = k_raw.view(B*R, C)
        v_flat = v_raw.view(B*R, C)
        k_sel = k_flat[idx_off].view(B, Q, P, K, C)        # (B, Q, P, K, C)
        v_sel = v_flat[idx_off].view(B, Q, P, K, C)

          # 新增: Gather邻居的位置和速度, 用于后续计算门控
        rad_pos_flat = rad_pos.view(B*R, 3)
        # --- FIX STARTS HERE ---
        rad_vel_flat = rad_vel.view(B*R, 2) # FIX: Change 1 to 2 for 2D velocity
        rad_pos_sel = rad_pos_flat[idx_off].view(B, Q, P, K, 3) # (B,Q,P,K,3)
        rad_vel_sel = rad_vel_flat[idx_off].view(B, Q, P, K, 2) # FIX: Change 1 to 2 for 2D velocity
        # --- FIX ENDS HERE ---
        
        # 由于P=1, 去掉多余的维度
        rad_pos_sel = rad_pos_sel.squeeze(2) # -> (B,Q,K,3)
        rad_vel_sel = rad_vel_sel.squeeze(2) # -> (B,Q,K,2)

        # 5. 新增: 在查询和其K个邻居之间计算门控(gate)
        # 准备用于广播计算的img_pos和img_vel
        img_pos_exp = img_pos.unsqueeze(2) # (B,Q,1,3)
        img_vel_exp = img_vel.unsqueeze(2) # (B,Q,1,2)

        if self.distance_gate_type == "xyz":
            dist = (img_pos_exp - rad_pos_sel).norm(dim=-1) # (B,Q,K)
        elif self.distance_gate_type == "xy_z":
            dxy = (img_pos_exp[..., :2] - rad_pos_sel[..., :2]).norm(dim=-1) # (B,Q,K)
            dz = (img_pos_exp[..., 2] - rad_pos_sel[..., 2]).abs() # (B,Q,K)
        
        # --- FIX STARTS HERE ---
        # FIX: Calculate the L2 norm of the velocity difference vector
        dv = (img_vel_exp - rad_vel_sel).norm(dim=-1) # (B,Q,K)
        # --- FIX ENDS HERE ---

        if self.use_gate:
            if self.distance_gate_type == "xyz":
                gate = self._distance_gate(dist, dv)
            elif self.distance_gate_type == "xy_z":
                gate = self._distance_gate_sep(dxy, dz, dv)
            else:
                raise ValueError(f"Invalid distance gate type: {self.distance_gate_type}")
            # gate的形状是 (B,Q,K)

        # 6. 多头拆分 (原步骤5)
        q = q_raw.view(B, Q, H, D_h).permute(0,2,1,3)       # (B,H,Q,D_h)
        # 把 P×K 展平为一维
        k = k_sel.view(B, Q, P*K, H, D_h).permute(0,3,1,2,4) # (B,H,Q,P*K,D_h)
        v = v_sel.view(B, Q, P*K, H, D_h).permute(0,3,1,2,4)

        # 7. 计算注意力, 应用门控并聚合 (原步骤6)
        attn_logits = (q.unsqueeze(3) @ k.transpose(-2,-1)).squeeze(3) * self.scale  # (B,H,Q,P*K)
        
        # 修改: 在这里应用门控
        if self.use_gate:
            # gate (B,Q,K) -> (B,1,Q,K) 以便广播到所有注意力头
            attn_logits = attn_logits + torch.log(gate.unsqueeze(1) + 1e-6)

        attn = torch.softmax(attn_logits, dim=-1)
        attn = self.dropout(attn)
        ctx = torch.einsum('bhqp,bhqpd->bhqd', attn, v)      # (B,H,Q,D_h)
        ctx = ctx.transpose(1,2).reshape(B, Q, C)            # (B,Q,C)

        return ctx

    # ------------------------------------------------------------------
    # main forward
    # ------------------------------------------------------------------
    def forward(
        self,
        img_feat:   torch.Tensor,              # (B,Q,D)
        radar_feat: torch.Tensor,              # (B,R,D_radar)
        img_xyz10:  Optional[torch.Tensor],    # (B,Q,10)
        radar_xyz10:Optional[torch.Tensor],    # (B,R,10)
        point_mask:Optional[torch.Tensor],    # (B,R,1)
    ):
        
        # 线性映射
        q_raw = self.q_proj(img_feat)          # (B,Q,D)
        k_raw = self.k_proj(radar_feat)        # (B,R,D)
        v_raw = self.v_proj(radar_feat)        # (B,R,D)

        # Attention
        if self.attn_type == "dot":
            q = q_raw.view(img_feat.size(0), img_feat.size(1), self.num_heads, -1).transpose(1, 2)
            k = k_raw.view(radar_feat.size(0), radar_feat.size(1), self.num_heads, -1).transpose(1, 2)
            v = v_raw.view(radar_feat.size(0), radar_feat.size(1), self.num_heads, -1).transpose(1, 2)
            context = self._forward_dot(q, k, v, img_xyz10, radar_xyz10, point_mask)
        elif self.attn_type == "dot_optimized":
            # 高效但對維度敏感的版本
            q = q_raw.view(img_feat.size(0), img_feat.size(1), self.num_heads, -1).transpose(1, 2)
            k = k_raw.view(radar_feat.size(0), radar_feat.size(1), self.num_heads, -1).transpose(1, 2)
            v = v_raw.view(radar_feat.size(0), radar_feat.size(1), self.num_heads, -1).transpose(1, 2)
            context = self._forward_topk_dot_gather(q, k, v, img_xyz10, radar_xyz10, point_mask)
        else:  # "deform"
            context = self._forward_deform(q_raw, k_raw, v_raw, img_xyz10, radar_xyz10)

        # 输出投影 + 融合
        context = self.out_proj(context)
        if self.fuse == "add":
            fused = self.norm(img_feat + self.dropout(context))
        else:  # "cat"
            fused = self.norm(self.cat_proj(torch.cat([img_feat, context], dim=-1)))
        return fused


class RadarPointSelfAttention(nn.Module):
    """
    Self‑attention 模块（带前馈、残差、LayerNorm）
    输入尺寸 : [B, N, 256]
    参数说明
    ----------
    embed_dim : 特征维度（默认为 256）
    num_heads : 多头数
    dropout   : dropout 概率
    """
    def __init__(self, embed_dim: int = 256, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()

        # 1) 多头自注意力；batch_first=True 方便直接输入 [B, N, C]
        self.attn = nn.MultiheadAttention(embed_dim, num_heads,
                                          dropout=dropout, batch_first=True)

        # 2) 后续 MLP / FFN
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim)
        )

        # 3) LayerNorm（两次残差之前各放一次）
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # 4) 可选：权重初始化（Xavier 更适合 Linear）
        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.MultiheadAttention):
                # PyTorch 已自带合适初始化，若需要可额外调参
                nn.init.xavier_uniform_(m.in_proj_weight)
                if m.in_proj_bias is not None:
                    nn.init.constant_(m.in_proj_bias, 0.)
                nn.init.xavier_uniform_(m.out_proj.weight)
                if m.out_proj.bias is not None:
                    nn.init.constant_(m.out_proj.bias, 0.)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None):
        """
        Parameters
        ----------
        x : Tensor [B, N, 256]
        key_padding_mask : Bool Tensor [B, N]，True 位置会被 mask 掉（可选）
        Returns
        -------
        out : Tensor [B, N, 256]   —— 经过两次残差的输出
        attn_weights : Tensor [B, N, N] —— 注意力权重（如果需要可返回）
        """
        # --- Self‑Attention sub‑layer ---
        attn_out, attn_weights = self.attn(
            query=x, key=x, value=x, key_padding_mask=key_padding_mask)  # [B, N, C]
        x = self.norm1(x + attn_out)  # 残差 + LN

        # --- Feed‑Forward sub‑layer ---
        ffn_out = self.ffn(x)         # [B, N, C]
        out = self.norm2(x + ffn_out) # 残差 + LN
        return out      # 若不需要 attn_weights，可以只返回 out

class HeightSoftGate(nn.Module):
    """
    Soft gate for height difference between query bbox and radar points.
    若某帧 N == 0（无雷达点），则返回全 1，等价于禁用本模块。
    """
    def __init__(self,
                 gate_type: str = "gaussian",
                 sigma_init: float = 0.2,
                 tau: float = 0.2,
                 kappa: float = 10.0,
                 min_gate: float = 0,
                 learnable_sigma: bool = True):
        super().__init__()
        assert gate_type in {"gaussian", "logistic"}
        self.gate_type  = gate_type
        self.min_gate   = min_gate
        self.tau        = tau
        self.kappa      = kappa
       
        if gate_type == "gaussian":
            if learnable_sigma:
                self.log_sigma = nn.Parameter(torch.log(torch.tensor(sigma_init)))
            else:
                self.register_buffer("sigma", torch.tensor(sigma_init))

    # 访问 (learnable / 固定) σ
    @property
    def sigma(self) -> torch.Tensor:
        if self.gate_type != "gaussian":
            raise AttributeError("only gaussian gate has sigma")
        if hasattr(self, "log_sigma"):
            return torch.exp(self.log_sigma)
        return self._buffers["sigma"]

    @torch.no_grad()  # 仅做简单计算，可省显存；若需反传删掉此行
    def forward(self,
                radar_z: torch.Tensor,      # (B, N) or (B, N, 1)
                query_bbox: torch.Tensor):  # (B, Q, 10)
        B, Q = query_bbox.shape[:2]

        # ------------------------------------------------------------
        # 0. 无雷达点 ⇒ gate = 1  (模块失效)
        # ------------------------------------------------------------
        if radar_z is None or radar_z.size(1) == 0:
            return torch.ones((B, Q), device=query_bbox.device, dtype=query_bbox.dtype)

        # ------------------------------------------------------------
        # 1. 取中心高度并归一化
        # ------------------------------------------------------------
        radar_z = radar_z.squeeze(-1)                        # (B, N)
        qry_z   = query_bbox[..., 2]                         # (B, Q)

   
        # ------------------------------------------------------------
        # 2. Δz_min
        # ------------------------------------------------------------
        diff = torch.abs(qry_z.unsqueeze(-1) - radar_z.unsqueeze(1))  # (B,Q,N)
        diff = torch.clamp(diff, min=0.0, max=1.0)
        delta_z_min, _ = diff.min(dim=-1)                    # (B, Q)
        # print(delta_z_min)

        # ------------------------------------------------------------
        # 3. Soft gate
        # ------------------------------------------------------------
        if self.gate_type == "gaussian":
            w = torch.exp(-0.5 * (delta_z_min / self.sigma) ** 2)  # (B,Q)
        else:  # logistic
            w = 1.0 / (1.0 + torch.exp(self.kappa * (delta_z_min - self.tau)))

        gate = self.min_gate + (1.0 - self.min_gate) * w           # (B,Q)
        # print(gate)
        return gate


class HeightHardGate(nn.Module):
    """
    基于归一化高度差的硬门控 (Hard Gate)。
    通过乘法作用于特征通道，当最小归一化高度差超过阈值时返回 0 (阻断)，否则返回 1 (通过)。
    若某帧 N == 0 (无雷达点)，则返回全 1，等价于禁用本模块。
    """
    def __init__(self,
                height_diff_threshold: float = 0.7,
                ):
        super().__init__()
        # 阈值，作用于 0-1 范围的归一化高度差
        self.height_diff_threshold = height_diff_threshold
      

    def forward(self,
                radar_z: torch.Tensor,      # (B, N) or (B, N, 1)
                query_bbox: torch.Tensor):  # (B, Q, 10)
        """
        返回一个形状为 (B, Q) 的乘法门控 (值为 0 或 1)。
        有效位置为 1，无效位置为 0。
        """
        B, Q = query_bbox.shape[:2]
        device = query_bbox.device

        # ------------------------------------------------------------
        # 0. 无雷达点 ⇒ gate = 1 (乘法门控，1代表无效果/通过)
        # ------------------------------------------------------------
        if radar_z is None or radar_z.size(1) == 0:
            return torch.ones((B, Q), device=device, dtype=query_bbox.dtype)

        # ------------------------------------------------------------
        # 1. 计算归一化的最小高度差
        # ------------------------------------------------------------
        radar_z = radar_z.squeeze(-1)           # (B, N)
        qry_z   = query_bbox[..., 2]            # (B, Q)

        # 计算每个 query 和所有 radar 点的高度差的绝对值
        diff = torch.abs(qry_z.unsqueeze(-1) - radar_z.unsqueeze(1))  # (B, Q, N)
        
        # !! 关键步骤：进行归一化和裁剪 !!
        diff_normalized = torch.clamp(diff , min=0.0, max=1.0)

        # 找到每个 query 对应的最小归一化高度差
        delta_z_min, _ = diff_normalized.min(dim=-1)       # (B, Q)

        # ------------------------------------------------------------
        # 2. 生成 Hard Gate (0/1)
        # ------------------------------------------------------------
        # 默认所有通道都通过 (值为1)
        gate = torch.ones_like(delta_z_min)

        # 找出最小归一化高度差超过阈值的位置
        is_invalid = delta_z_min > self.height_diff_threshold

        # 将这些无效位置的 gate 值设为 0 (阻断)
        gate[is_invalid] = 0.0

        return gate
