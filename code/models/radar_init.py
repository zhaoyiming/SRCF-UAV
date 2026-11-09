import torch
import torch.nn as nn
import math
from typing import List

def normalize_points(bboxes: torch.Tensor, pc_range: List[float], filter_border=False):

        xyz = bboxes[..., 0:3].clone()
        if pc_range is not None:
            xyz[..., 0] = (xyz[..., 0] - pc_range[0]) / (pc_range[3] - pc_range[0])
            xyz[..., 1] = (xyz[..., 1] - pc_range[1]) / (pc_range[4] - pc_range[1])
            xyz[..., 2] = (xyz[..., 2] - pc_range[2]) / (pc_range[5] - pc_range[2])
        
        bboxes[..., :3]=xyz
        return bboxes


class QueryInitializer(nn.Module):
    """
    两种 radar 点“密集化”策略：
      - learnable: 用 MLP 为每个雷达点预测 k_per_point 个偏移（在归一化坐标下直接学习偏移）
      - sphere:    在每个雷达点为球心、半径 radius（米级）取球面均匀采样 k_per_point 个方向，
                   但在归一化空间考虑不同轴的尺度
    输入 radar_xyz 为归一化后的 [0,1] 坐标，extent 为归一化前各轴的长度 [dx,dy,dz]
    返回 jittered 查询点：归一化 [0,1]
    """

    def __init__(
        self,
        strategy: str = 'sphere',
        k_per_point: int = 6,
        radius: float = 0.5,
        extent: float=[]
    ):
        """
        Args:
            strategy: 'learnable' 或 'sphere'
            k_per_point: 每个雷达点扩增出的点数
            radius: 球面采样半径（米）
            extent: 归一化前各轴的空间范围 (dx, dy, dz)
        """
        super().__init__()
        assert strategy in ('learnable', 'sphere'), "strategy 必须是 'learnable' 或 'sphere'"
        self.strategy = strategy
        self.k = k_per_point
        self.radius = radius
        # extent: 在归一化前，x/y/z 轴的物理长度
        self.register_buffer('extent', torch.tensor(extent, dtype=torch.float32))  # (3,)

        if strategy == 'learnable':
            self.mlp = nn.Sequential(
                nn.Linear(3, 64),
                nn.ReLU(),
                nn.Linear(64, k_per_point * 3)
            )

    def sample_on_sphere(self, B: int, R: int, device: torch.device) -> torch.Tensor:
        # 在单位球面上随机采样 k_per_point 个方向
        u   = torch.rand(B, R, self.k, device=device) * 2 - 1           # cosθ ∈ [-1,1]
        phi = torch.rand(B, R, self.k, device=device) * 2 * math.pi     # φ ∈ [0,2π)
        x   = torch.sqrt(1 - u**2) * torch.cos(phi)
        y   = torch.sqrt(1 - u**2) * torch.sin(phi)
        z   = u
        return torch.stack([x, y, z], dim=-1)  # (B, R, k, 3)

    def forward(self, radar_xyz: torch.Tensor) -> torch.Tensor:
        """
        Args:
            radar_xyz: Tensor, shape (B, R, 3), 归一化到 [0,1] 的雷达点
        Returns:
            jittered_xyz: Tensor, shape (B, R*k_per_point, 3)，归一化 [0,1] 的扩增点
        """
        B, R, _ = radar_xyz.shape
        dev = radar_xyz.device
        base = radar_xyz.unsqueeze(2)  # (B, R, 1, 3)

        if self.strategy == 'learnable':
            off = self.mlp(radar_xyz)             # (B, R, k*3)
            off = off.view(B, R, self.k, 3)      # (B, R, k, 3)
            jitter = base + off                  # 在归一化空间直接加偏移
        else:
            # 1) 单位球面方向
            dirs = self.sample_on_sphere(B, R, dev)  # (B, R, k, 3)
            # 2) 计算各轴上的归一化半径：radius_norm = radius_m / extent_m
            #    extent 存储的是归一化前的各轴长度
            radius_norm = self.radius / self.extent    # (3,)
            # 3) 在归一化空间缩放并加到 base
            jitter = base + dirs * radius_norm.view(1, 1, 1, 3)

        # 展平成 (B, R*k, 3)
        return jitter.reshape(B, R * self.k, 3)
    
class RadarQueryInitializer(nn.Module):
    # __init__ 和 QueryInitializer 保持不变
    def __init__(self, num_query=900,
                 init_strategy='sphere',
                 k_per_point=4,
                 sphere_radius=0.5):
        super().__init__()
        self.num_query = num_query
        self.query_initializer = QueryInitializer(
            strategy=init_strategy,
            k_per_point=k_per_point,
            radius=sphere_radius,
            extent=[51.2, 51.2, 10.0]
        )
        if init_strategy == 'learnable':
            self.add_module('offset_mlp', self.query_initializer.mlp)
        self.point_cloud_range=[-25.6, -25.6, 0.0, 25.6, 25.6, 10]

    def assign_radarpoints(self, radar_data, query_bbox, eps=1e-4):
        B, R, T = radar_data.shape
        dev = radar_data.device
        if R == 0:
            return query_bbox
        assert R<100, "R should be less than 100"

        # print(radar_data)

        # radar_xyz = radar_data[..., :3]
        # radar_vel = radar_data[..., 3:]
        # C_vel = radar_vel.shape[-1]
        # k = self.query_initializer.k
        # jitter_pts = self.query_initializer(radar_xyz)
        # vel_expanded = radar_vel.unsqueeze(2).expand(B, R, k, C_vel)
        # jitter_vel = vel_expanded.reshape(B, R * k, C_vel)
        # jitter_data = torch.cat([jitter_pts, jitter_vel], dim=-1)
        # all_candidates = jitter_data
        all_candidates=radar_data
        all_candidates = normalize_points(all_candidates, self.point_cloud_range)
        all_candidates=all_candidates.clamp(eps, 1-eps)
    
        Nc = all_candidates.shape[1]
        start, end = 11, 899
        allowed_idx = torch.arange(start, end, device=dev)   
        dest_indices = allowed_idx[:Nc].unsqueeze(0).expand(B, -1)   # shape = (B, Nc)
        query_bbox[0, dest_indices, :3] = all_candidates[0, :, :3]
        # query_bbox[0, dest_indices, -1] = all_candidates[0, :, -3]
        query_bbox[0, dest_indices, 8:10] = all_candidates[0, :, -2:]
        
        zero_bbox = torch.zeros([query_bbox.size()[0], query_bbox.size()[1], T], device=query_bbox.device)
        zero_bbox[0, dest_indices, :] = all_candidates[0, :, :]
  

        assert query_bbox[0, dest_indices, :3].all()>=-1 and query_bbox[0, dest_indices, :3].all()<=1 and query_bbox[0, dest_indices, :3].all()!=0, "query_bbox[0, dest_indices, :3] should be all greater than 0 and less than 1"
        return query_bbox, zero_bbox

            


    def assign_queries(self,
                       radar_data,  # (B, R, D)
                       query_bbox   # (B, Q, D_query)
                       ):
        B, R, _ = radar_data.shape
        dev = radar_data.device
        
        if R == 0:
            return query_bbox

        # 1. & 2. 生成候选池 (这部分逻辑和之前一样)
        radar_xyz = radar_data[..., :3]
        radar_vel = radar_data[..., 3:]
        C_vel = radar_vel.shape[-1]
        k = self.query_initializer.k

        jitter_pts = self.query_initializer(radar_xyz)
        vel_expanded = radar_vel.unsqueeze(2).expand(B, R, k, C_vel)
        jitter_vel = vel_expanded.reshape(B, R * k, C_vel)
        
        jitter_data = torch.cat([jitter_pts, jitter_vel], dim=-1)
        # all_candidates = torch.cat([radar_data, jitter_data], dim=1) # (B, Nc,
        #  D_feat)
        all_candidates=radar_data
        all_candidates = normalize_points(all_candidates, self.point_cloud_range)
        # print(all_candidates)
        # exit()
        
        
        # 3. 核心修正：为 all_candidates 找到随机的目标位置
        Nc = all_candidates.shape[1]
        
        # 从 num_query 个槽位中，为每个批次样本随机选择 Nc 个目标索引
        # 我们使用 rand().topk() 技巧来实现批量的随机不重复抽样
        start, end = 11, 899
        allowed_idx = torch.arange(start, end, device=dev)   
        rand_scores = torch.rand(B, allowed_idx.numel(), device=dev)
        _, topk = rand_scores.topk(Nc, dim=1) # (B, Nc)
      
        dest_indices = allowed_idx[topk]                             # shape = (B, Nc)

        # 4. 使用高级索引执行赋值 (Scatter Operation)
        # 我们需要一个批次索引来配合 dest_indices
        batch_indices = torch.arange(B, device=dev).unsqueeze(1).expand(-1, Nc) 
      
      
        query_bbox[batch_indices, dest_indices, :3] = all_candidates[:, :, :3]
        query_bbox[batch_indices, dest_indices, 8:10] = all_candidates[:, :, 3:]

        return query_bbox



@torch.no_grad()
def init_xy_900_from_prev_batch(
    prev_xy: torch.Tensor,                  # [B, 256, 2], sigmoid后 => in [0,1]
    grid_size: int = 30,                    # 30x30 => 900
    use_scores: bool = False,
    prev_scores: torch.Tensor = None,# [B, 256], 可选
    sigma: float = 0.15,                    # 形变半径（0~1坐标系）
    alpha_max: float = 0.6,                 # 最大吸引强度
    score_power: float = 1.0,               # 分数锐化/平滑（>1 更偏向高分）
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    返回 init_xy: [B, 900, 2]，均为 0~1 归一化（sigmoid坐标系）
    - use_scores=False: 最近邻吸引（hard assignment），速度快、稳定
    - use_scores=True : 分数加权软吸引（soft assignment），更平滑，更能抑制低分query
    """
    assert prev_xy.dim() == 3 and prev_xy.size(-1) == 2, "prev_xy must be [B, N, 2]"
    B, N, _ = prev_xy.shape
    assert N == 256, f"expected N=256, got N={N}"

    device, dtype = prev_xy.device, prev_xy.dtype

    # ---- 0) 确保输入在[0,1]
    prev_xy = prev_xy.clamp(0.0, 1.0)

    # ---- 1) base grid centers: [Q,2] where Q=grid_size^2 (=900)
    x = torch.arange(grid_size, device=device, dtype=dtype)
    y = torch.arange(grid_size, device=device, dtype=dtype)
    xx, yy = torch.meshgrid(x, y, indexing="ij")         # [G,G]
    grid = torch.stack([xx, yy], dim=-1)                 # [G,G,2]
    grid = (grid + 0.5) / grid_size                      # cell center in (0,1)
    grid = grid.reshape(-1, 2)                           # [Q,2]
    Q = grid.size(0)

    # ---- 2) pairwise squared distances: [B,Q,N]
    # grid -> [1,Q,1,2], prev -> [B,1,N,2]
    diff  = grid[None, :, None, :] - prev_xy[:, None, :, :]   # [B,Q,N,2]
    dist2 = (diff * diff).sum(dim=-1)                         # [B,Q,N]

    # ---- 3) alpha based on nearest distance (避免远处塌缩)
    d2_min, nn_idx = dist2.min(dim=2)                     # [B,Q], [B,Q]
    inv = 1.0 / (2.0 * sigma * sigma + eps)
    alpha = alpha_max * torch.exp(-d2_min * inv)           # [B,Q]
    alpha = alpha.unsqueeze(-1)                            # [B,Q,1]

    # ---- 4) target points
    if (not use_scores) or (prev_scores is None):
        # (A) hard: nearest neighbor gather -> [B,Q,2]
        b_idx = torch.arange(B, device=device)[:, None]    # [B,1]
        target_xy = prev_xy[b_idx, nn_idx]                 # advanced indexing, [B,Q,2]
    else:
        # (B) soft: score-weighted assignment
        assert prev_scores.dim() == 2 and prev_scores.shape[:2] == (B, N), \
            "prev_scores must be [B, N] aligned with prev_xy"

        s = prev_scores.to(device=device, dtype=dtype).clamp(0.0, 1.0)  # [B,N]
        if score_power != 1.0:
            s = s.pow(score_power)

        # w_bqn ∝ s_bn * exp(-dist2_bqn/(2*sigma^2))
        w = torch.exp(-dist2 * inv) * s[:, None, :]        # [B,Q,N]
        w = w / (w.sum(dim=2, keepdim=True) + eps)         # normalize over N

        # target_xy[b,q,:] = Σ_n w[b,q,n] * prev_xy[b,n,:]
        target_xy = torch.einsum("bqn,bnd->bqd", w, prev_xy)  # [B,Q,2]

    # ---- 5) warp grid toward target (grid broadcast to [B,Q,2])
    init_xy = (1.0 - alpha) * grid[None, :, :] + alpha * target_xy  # [B,Q,2]

    # ---- 6) ensure sigmoid space valid
    init_xy = init_xy.clamp(0.0, 1.0)
    return init_xy


@torch.no_grad()
def init_xy_900_from_prev(
    prev_xy: torch.Tensor,            # [256, 2], sigmoid后 => in [0,1]
    grid_size: int = 30,              # 30x30 => 900
    use_scores: bool = False,
    prev_scores: None = None,  # [256], 可选
    sigma: float = 0.15,              # 形变半径（0~1坐标系）
    alpha_max: float = 0.6,           # 最大吸引强度
    score_power: float = 1.0,         # 分数的锐化/平滑指数（>1 更偏向高分）
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    返回 init_xy: [900, 2]，均为 0~1 归一化（sigmoid坐标系）
    - use_scores=False: 最近邻吸引（硬分配），速度快、稳定
    - use_scores=True : 分数加权的软吸引（soft assignment），更平滑，更能抑制低分query
    """
    assert prev_xy.dim() == 2 and prev_xy.size(-1) == 2, "prev_xy must be [N,2]"
    device, dtype = prev_xy.device, prev_xy.dtype

    # ---- 0) 确保输入在[0,1]（sigmoid后通常已满足，这里做防御性clamp）
    prev_xy = prev_xy.clamp(0.0, 1.0)

    # ---- 1) base grid centers: [900,2]
    x = torch.arange(grid_size, device=device, dtype=dtype)
    y = torch.arange(grid_size, device=device, dtype=dtype)
    xx, yy = torch.meshgrid(x, y, indexing="ij")         # [G,G]
    grid = torch.stack([xx, yy], dim=-1)                 # [G,G,2]
    grid = (grid + 0.5) / grid_size                      # cell center in (0,1)
    grid = grid.reshape(-1, 2)                           # [900,2]

    # ---- 2) pairwise squared distances: [900,256]
    diff  = grid[:, None, :] - prev_xy[None, :, :]       # [900,256,2]
    dist2 = (diff * diff).sum(dim=-1)                    # [900,256]

    # ---- 3) attraction strength alpha_j based on min distance (避免远处塌缩)
    d2_min, nn_idx = dist2.min(dim=1)                    # [900], [900]
    inv = 1.0 / (2.0 * sigma * sigma + eps)
    alpha = alpha_max * torch.exp(-d2_min * inv)         # [900]
    alpha = alpha.unsqueeze(1)                           # [900,1]

    # ---- 4) compute target points to attract to
    if (not use_scores) or (prev_scores is None):
        # (A) no score: nearest neighbor (hard)
        target_xy = prev_xy[nn_idx]                      # [900,2]
    else:
        # (B) with score: score-weighted soft assignment
        assert prev_scores.dim() == 1 and prev_scores.numel() == prev_xy.size(0), \
            "prev_scores must be [N] aligned with prev_xy"

        # score处理：clamp到[0,1]，可用power调节偏好
        s = prev_scores.to(device=device, dtype=dtype).clamp(0.0, 1.0)
        if score_power != 1.0:
            s = s.pow(score_power)

        # 权重 w_ji ∝ s_i * exp(-||g_j-p_i||^2/(2*sigma^2))
        w = torch.exp(-dist2 * inv) * s.unsqueeze(0)     # [900,256]
        w = w / (w.sum(dim=1, keepdim=True) + eps)       # normalize
        target_xy = w @ prev_xy                          # [900,2]

    # ---- 5) warp grid toward target
    init_xy = (1.0 - alpha) * grid + alpha * target_xy   # [900,2]

    # ---- 6) ensure sigmoid space valid
    init_xy = init_xy.clamp(0.0, 1.0)
    return init_xy