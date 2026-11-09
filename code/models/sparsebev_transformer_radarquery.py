from re import T
import torch
from torch._functorch.autograd_function import vmapify_autograd_function
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from mmcv.runner import BaseModule
from mmcv.cnn import bias_init_with_prob
from mmcv.cnn.bricks.transformer import MultiheadAttention, FFN
from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttention
from mmdet.models.utils.builder import TRANSFORMER
from .bbox.utils import decode_bbox
from .utils import inverse_sigmoid, DUMP
from .sparsebev_sampling import sampling_4d, make_sample_points
from .checkpoint import checkpoint as cp
from .csrc.wrapper import MSMV_CUDA
from mmcv.runner import get_dist_info
from .radar_sampling import RangeAdaptiveRadarAggregation, RadarPointSelfAttention, HeightSoftGate, HeightHardGate
from .query_stack import QueryStack
import math

class MHA_LogitScale(nn.Module):
    """
    原生 MultiheadAttention + dt 作为“logits缩放/温度”：
      logits = (QK^T)/sqrt(d) * s(dt)
    实现方式：对 query/key 输入做同尺度缩放（不需要 attn_mask）。
    s(dt)=exp(-dt/tau)  (per-sample 标量，稳定有效)
    """
    def __init__(self, embed_dim, num_heads, dropout=0.0, tau_init=0.3, tau_min=1e-4, batch_first=True):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=batch_first)
        inv_softplus = lambda x: math.log(math.exp(x) - 1.0)
        self.tau_param = nn.Parameter(torch.tensor(inv_softplus(float(tau_init))))
        self.tau_min = float(tau_min)

    def forward(self, query, key, value, dt, key_padding_mask=None):
        B = query.size(0)
        dt = dt.to(dtype=query.dtype, device=query.device)
        dt = dt.clamp(0.0, 1.0)              # 非 inplace
        tau = F.softplus(self.tau_param) + self.tau_min

        s = torch.exp(-(dt / tau)).view(B, 1, 1)  # [B,1,1]
        # 让 logits 缩放：等价于 q 或 k 其中一个缩放，这里缩放 query（更直观）
        q = query * s

        out, _ = self.mha(q, key, value, key_padding_mask=key_padding_mask, need_weights=False)
        return out

class KVDecayMHA(nn.Module):
    """
    Use nn.MultiheadAttention. Apply decay to key/value BEFORE MHA:
      g(dt) = exp(-dt/tau)  (scalar per sample)
    """
    def __init__(self, embed_dim, num_heads, dropout=0.0,  lambda_init=0.3, lambda_min=1e-4, batch_first=True):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=batch_first)

        inv_softplus = lambda x: math.log(math.exp(x) - 1.0)
        init = inv_softplus(float(lambda_init))
        self.lambda_param = nn.Parameter(torch.tensor(float(init)))
        self.lambda_min = float(lambda_min)

    def forward(self, query, key, value, dt, key_padding_mask=None):
        B, _, _ = query.shape
        dt = dt.to(dtype=query.dtype, device=query.device)
        dt = torch.clamp(dt, min=0.0, max=1.0)
        tau = F.softplus(self.lambda_param).to(dtype=query.dtype, device=query.device) + self.lambda_min

        g = torch.exp(-(dt / tau)).view(B, 1, 1)  # [B,1,1]
        key = key * g
        value = value * g

        out, _ = self.mha(query, key, value, key_padding_mask=key_padding_mask, need_weights=False)
        return out

# class KVDecayMHA(nn.Module):
#     """
#     Apply displacement-aware decay to key/value BEFORE MHA (per historical token):
#       speed_k = sqrt(vx_k^2 + vy_k^2)
#       d_k = speed_k * dt
#       g_k = exp(- d_k / lambda)
#     Shapes:
#       query: [B, Lq, D]    (e.g., [B,900,256])
#       key/value: [B, Lk, D] (e.g., [B,256,256])  -- historical memory
#       v_xy: [B, Lk, 2]     -- velocity for each historical token
#       dt: [B] or [B,1]
#     """
#     def __init__(self, embed_dim, num_heads, dropout=0.0,
#                  lambda_init=1.0, lambda_min=1e-4, batch_first=True, eps=1e-12):
#         super().__init__()
#         self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=batch_first)

#         inv_softplus = lambda x: math.log(math.exp(x) - 1.0)
#         self.lambda_param = nn.Parameter(torch.tensor(inv_softplus(float(lambda_init))))
#         self.lambda_min = float(lambda_min)
#         self.eps = float(eps)

#     def forward(self, query, key, value, dt, v_xy, key_padding_mask=None, debug=False):
#         B, Lq, D = query.shape
#         _, Lk, Dk = key.shape
#         if Dk != D:
#             raise ValueError(f"embed_dim mismatch: query D={D}, key D={Dk}")

#         device, dtype = query.device, query.dtype

#         # dt: [B] or [B,1] -> [B,1] -> expand to [B,Lk]
#         dt = dt.to(device=device, dtype=dtype)
#         dt = torch.clamp(dt, min=0.0)
#         if dt.dim() == 1:
#             dt = dt[:, None]                  # [B,1]
#         elif dt.dim() == 2 and dt.shape[1] == 1:
#             pass                               # [B,1]
#         else:
#             raise ValueError(f"dt must be [B] or [B,1], got {tuple(dt.shape)}")
#         dt_tok = dt.expand(B, Lk)              # [B,Lk]

#         # v_xy: [B,Lk,2]
#         v_xy = v_xy.to(device=device, dtype=dtype)
#         if v_xy.dim() != 3 or v_xy.shape[0] != B or v_xy.shape[1] != Lk or v_xy.shape[2] != 2:
#             raise ValueError(f"v_xy must be [B,Lk,2]=[{B},{Lk},2], got {tuple(v_xy.shape)}")

#         speed = torch.sqrt(v_xy[..., 0] ** 2 + v_xy[..., 1] ** 2 + self.eps)  # [B,Lk]

#         lam = F.softplus(self.lambda_param).to(device=device, dtype=dtype) + self.lambda_min
#         d = speed * dt_tok                              # [B,Lk]
#         g = torch.exp(-(d / lam)).unsqueeze(-1)         # [B,Lk,1]

#         if debug:
#             print("speed:", speed)
#         #     print("dt:", dt)
#         #     print("d:", d)
#         #     # print("g:", g)
#             print("g range:", g.min().item(), g.max().item())
#         #     exit(0)
#             # print("query:", query.shape, "key:", key.shape, "v_xy:", v_xy.shape)
#             # print("dt:", dt.shape, "dt_tok:", dt_tok.shape)
#             # print("speed:", speed.shape, "d:", d.shape, "g:", g.shape)
#             # print("g range:", g.min().item(), g.max().item())

#         key = key * g
#         value = value * g

#         out, _ = self.mha(query, key, value, key_padding_mask=key_padding_mask, need_weights=False)
#         return out




@TRANSFORMER.register_module()
class SparseBEVTransformer_radarquery(BaseModule):
    def __init__(self, embed_dims, num_frames=8, num_points=4, num_layers=6, num_levels=4, num_classes=10, code_size=10, pc_range=[], num_cams=1,
                 use_query_stack=False, num_temp_queries=256, num_query=900, temporal_decay_tau=0.3,
                init_cfg=None):
        assert init_cfg is None, 'To prevent abnormal initialization ' \
                            'behavior, init_cfg is not allowed to be set'
        super(SparseBEVTransformer_radarquery, self).__init__(init_cfg=init_cfg)
        # print(embed_dims, num_frames, num_points, num_layers, num_levels, num_classes, code_size, pc_range, num_cams)
        # 256 4 8 5 4 1 10 [-25.6, -25.6, 0.0, 25.6, 25.6, 10] 1

        self.embed_dims = embed_dims
        self.pc_range = pc_range
        self.decoder = SparseBEVTransformerDecoder(embed_dims, num_frames, num_points, num_layers, num_levels, num_classes, code_size, pc_range=pc_range, num_cams=num_cams, use_query_stack=use_query_stack, num_temp_queries=num_temp_queries, num_query=num_query, temporal_decay_tau=temporal_decay_tau)

    @torch.no_grad()
    def init_weights(self):
        self.decoder.init_weights()

    def forward(self, query_bbox, query_feat, radar_query_bbox, radar_query_feat, mlvl_feats, point_feature, attn_mask, img_metas, vel_bbox, point_mask, query_stack=None):
        cls_scores, bbox_preds = self.decoder(query_bbox, query_feat, radar_query_bbox, radar_query_feat, mlvl_feats, point_feature, attn_mask, img_metas, vel_bbox, point_mask, query_stack=query_stack)

        if self.training and (not bool(torch.isfinite(cls_scores).all())
                              or not bool(torch.isfinite(bbox_preds).all())):
            raise FloatingPointError('Non-finite decoder outputs before nan_to_num; training stopped.')
        cls_scores = torch.nan_to_num(cls_scores)
        bbox_preds = torch.nan_to_num(bbox_preds)

        return cls_scores, bbox_preds


class SparseBEVTransformerDecoder(BaseModule):
    def __init__(self, embed_dims, num_frames=8, num_points=4, num_layers=6, num_levels=4, num_classes=10, code_size=10, pc_range=[], num_cams=1,
                 num_temp_queries=256, num_query=900, use_query_stack=False, temporal_decay_tau=1.0,
                 init_cfg=None):

        super(SparseBEVTransformerDecoder, self).__init__(init_cfg)
        self.num_layers = num_layers
        self.pc_range = pc_range
        self.decoder_layer = SparseBEVTransformerDecoderLayer(
            embed_dims, num_frames, num_points, num_levels, num_classes, code_size, pc_range=pc_range, num_cams=num_cams, temporal_decay_tau=temporal_decay_tau
        )
        self.num_cams = num_cams


    @torch.no_grad()
    def init_weights(self):
        self.decoder_layer.init_weights()

    def forward(self, query_bbox, query_feat, radar_query_bbox, radar_query_feat, mlvl_feats, point_feature, attn_mask, img_metas, vel_bbox, point_mask, query_stack=None):
        cls_scores, bbox_preds = [], []

        # calculate time difference according to timestamps
        timestamps = np.array([m['img_timestamp'] for m in img_metas], dtype=np.float64)
        timestamps = np.reshape(timestamps, [query_bbox.shape[0], -1, self.num_cams])
        time_diff = timestamps[:, :1, :] - timestamps
        time_diff = np.mean(time_diff, axis=-1).astype(np.float32)  # [B, F]
        time_diff = torch.from_numpy(time_diff).to(query_bbox.device)  # [B, F]
        img_metas[0]['time_diff'] = time_diff
        lidar2img = np.asarray([m['lidar2img'] for m in img_metas]).astype(np.float32)
        lidar2img = torch.from_numpy(lidar2img).to(query_bbox.device)  # [B, N, 4, 4]
        img_metas[0]['lidar2img'] = lidar2img

        
        # group image features in advance for sampling, see `sampling_4d` for more details
        for lvl, feat in enumerate(mlvl_feats):
            B, TN, GC, H, W = feat.shape  # [B, TN, GC, H, W]
            N, T, G, C = self.num_cams, TN // self.num_cams, 4, GC // 4
            feat = feat.reshape(B, T, N, G, C, H, W)

            if MSMV_CUDA:  # Our CUDA operator requires channel_last
                feat = feat.permute(0, 1, 3, 2, 5, 6, 4)  # [B, T, G, N, H, W, C]
                feat = feat.reshape(B*T*G, N, H, W, C)
            else:  # Torch's grid_sample requires channel_first
                feat = feat.permute(0, 1, 3, 4, 2, 5, 6)  # [B, T, G, C, N, H, W]
                feat = feat.reshape(B*T*G, C, N, H, W)

            mlvl_feats[lvl] = feat.contiguous()

        for i in range(self.num_layers):
            if get_dist_info()[0] == 0:
                DUMP.stage_count = i

            query_feat, cls_score, bbox_pred = self.decoder_layer(
               i, query_bbox, query_feat, radar_query_bbox, radar_query_feat, mlvl_feats,
               point_feature, attn_mask, img_metas, vel_bbox, point_mask,
               query_stack=query_stack
            )
            query_bbox = bbox_pred.clone().detach()

            cls_scores.append(cls_score)
            bbox_preds.append(bbox_pred)

            # ========== QueryStack: 更新queries（在layer处理后，根据配置的update_layers）=========
            if query_stack and i in query_stack.update_stage and not query_stack.individual_init and query_stack.prev_flag:
                # print("--------------------------------update query stack--------------------------------")
                query_feat, query_bbox = query_stack.update(
                    query_feature=query_feat, query_bbox=query_bbox, confidence=cls_scores[-1]
                )

        cls_scores = torch.stack(cls_scores)
        bbox_preds = torch.stack(bbox_preds)

        # ========== QueryStack: 缓存当前帧queries供下一帧使用 ==========
        if query_stack is not None:
            query_stack.cache(query_feat, query_bbox, cls_scores[-1], img_metas)
        return cls_scores, bbox_preds


class SparseBEVTransformerDecoderLayer(BaseModule):
    def __init__(self, embed_dims, num_frames=8, num_points=4, num_levels=4, num_classes=10, code_size=10, num_cls_fcs=2, num_reg_fcs=2, pc_range=[], init_cfg=None, num_cams=1, temporal_decay_tau=1.0):
        super(SparseBEVTransformerDecoderLayer, self).__init__(init_cfg)

        self.embed_dims = embed_dims
        self.num_classes = num_classes
        self.code_size = code_size
        self.pc_range = pc_range

        self.position_encoder = nn.Sequential(
            nn.Linear(3, self.embed_dims), 
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
        )

        self.self_attn = SparseBEVSelfAttention(embed_dims, num_heads=8, dropout=0.1, pc_range=pc_range)
        self.sampling = SparseBEVSampling(embed_dims, num_frames=num_frames, num_groups=4, num_points=num_points, num_levels=num_levels, pc_range=pc_range, num_cams=num_cams)
        self.mixing = AdaptiveMixing(in_dim=embed_dims, in_points=num_points * num_frames, n_groups=4, out_points=128)
        self.ffn = FFN(embed_dims, feedforward_channels=512, ffn_drop=0.1)

        self.norm1 = nn.LayerNorm(embed_dims)
        self.norm2 = nn.LayerNorm(embed_dims)
        self.norm3 = nn.LayerNorm(embed_dims)
   
    
        self.radar_norm=nn.LayerNorm(self.embed_dims)
        self.rar = RangeAdaptiveRadarAggregation()
        self.radar_self_attn = RadarPointSelfAttention(embed_dim=self.embed_dims, num_heads=8, dropout=0.1)
        self.radar_query_pos_encoder = nn.Sequential(
            nn.Linear(3, self.embed_dims), 
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
        )
        self.height_soft_gate = HeightSoftGate()
        self.height_gate_norm=nn.LayerNorm(self.embed_dims)
        self.height_hard_gate=HeightHardGate()

        # 时序注意力衰减参数
        self.temporal_decay_tau = temporal_decay_tau
        self.temp_gate_mlp = torch.nn.Sequential(
            torch.nn.Linear(2 * self.embed_dims + 1, 128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(128, 1),
        )
        self.kv_decay_mha = KVDecayMHA(embed_dims, num_heads=8, dropout=0.1, lambda_init=1, lambda_min=1e-4, batch_first=True)
        self.temp_attn = MultiheadAttention(embed_dims, num_heads=8, dropout=0.1, batch_first=True)
        self.norm_temp = nn.LayerNorm(embed_dims)
        # self.temp_attn_time = MHA_LogitScale(embed_dim=embed_dims, num_heads=8, dropout=0.1, tau_init=0.3)


        cls_branch = []
        for _ in range(num_cls_fcs):
            cls_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(nn.Linear(self.embed_dims, self.num_classes))
        self.cls_branch = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(num_reg_fcs):
            reg_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU(inplace=True))
        reg_branch.append(nn.Linear(self.embed_dims, self.code_size))
        self.reg_branch = nn.Sequential(*reg_branch)

        radar_query_bbox_refine_branch = []
        for _ in range(num_reg_fcs):
            radar_query_bbox_refine_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            radar_query_bbox_refine_branch.append(nn.ReLU(inplace=True))
        radar_query_bbox_refine_branch.append(nn.Linear(self.embed_dims, self.code_size))
        self.radar_query_bbox_refine_branch = nn.Sequential(*radar_query_bbox_refine_branch)

    @torch.no_grad()
    def init_weights(self):
        self.self_attn.init_weights()
        self.sampling.init_weights()
        self.mixing.init_weights()
        


        bias_init = bias_init_with_prob(0.01)
        nn.init.constant_(self.cls_branch[-1].bias, bias_init)

    def refine_bbox(self, bbox_proposal, bbox_delta):
        xyz = inverse_sigmoid(bbox_proposal[..., 0:3])
        xyz_delta = bbox_delta[..., 0:3]
        xyz_new = torch.sigmoid(xyz_delta + xyz)

        return torch.cat([xyz_new, bbox_delta[..., 3:]], dim=-1)
    
    def radar_query_bbox_refine(self, radar_query_bbox, radar_query_bbox_delta):
        xyz=inverse_sigmoid(radar_query_bbox[..., 0:3])
        xyz_delta=radar_query_bbox_delta[..., 0:3]
        xyz_new=torch.sigmoid(xyz_delta+xyz)
        return torch.cat([xyz_new, radar_query_bbox[..., 3:]], dim=-1)

   

    def forward(self, layer_index, query_bbox, query_feat, radar_query_bbox, radar_query_feat, mlvl_feats, point_feature, attn_mask, img_metas, vel_bbox, point_mask, query_stack=None):
        """
        Args:
            layer_index: decoder层索引
            query_bbox: [B, Q, 10] [cx, cy, cz, w, h, d, rot.sin, rot.cos, vx, vy]
            query_feat: [B, Q, embed_dims] - 当前帧query特征
            temp_query_feat: [B, num_temp_queries, embed_dims] or None - 历史帧query特征（仅在时序层传入）
        """
        query_pos = self.position_encoder(query_bbox[..., :3])
        query_feat = query_feat + query_pos
        query_feat = self.norm1(self.self_attn(query_bbox, query_feat, attn_mask))
        

        # print(query_stack.prev_flag, query_stack.fusion_stage, layer_index)
        if (query_stack is not None
            and layer_index in query_stack.fusion_stage
            and query_stack.prev_flag):
            # print("--------------------------------fusion query stack--------------------------------", layer_index)

            temp_query_feat = query_stack.get_temp_query_feature()   # [B, K, C]
            temp_query_bbox = query_stack.get_temp_query_bbox()[:,:,-2:]      # [B, K, 10]
            time_interval = query_stack.time_interval                # [B]
            mask = query_stack.mask.clone()                                  # [B] bool

            attn_out = self.kv_decay_mha(
                query_feat,          # [B, Q, C]
                temp_query_feat,     # [B, K, C]
                temp_query_feat,     # [B, K, C]
                time_interval,        # [B]
                # temp_query_bbox      # [B, K, 10]
            )
            # attn_out = self.temp_attn_time(
            #     query_feat,          # [B, Q, C]
            #     temp_query_feat,     # [B, K, C]
            #     temp_query_feat,     # [B, K, C]
            #     time_interval        # [B]
            # )
            # attn_out= self.temp_attn(query_feat, temp_query_feat, temp_query_feat)[0]
            m = mask.to(dtype=query_feat.dtype, device=query_feat.device).view(-1, 1, 1)
            query_feat = self.norm_temp(query_feat + m * attn_out)



                    
        
        # ========== 特征采样和融合 ==========
        sampled_feat = self.sampling(query_bbox, query_feat, mlvl_feats, img_metas)
        query_feat = self.norm2(self.mixing(sampled_feat, query_feat))


        radar_query_pos=self.radar_query_pos_encoder(radar_query_bbox[...,:3])
        radar_query_feat=radar_query_feat+radar_query_pos
        query_feat=self.rar(query_feat, radar_query_feat, query_bbox[...,:3], radar_query_bbox[...,:3], None)
        radar_query_feat=self.radar_self_attn(radar_query_feat)
        radar_query_bbox_delta=self.radar_query_bbox_refine_branch(radar_query_feat)
        radar_query_bbox=self.radar_query_bbox_refine(radar_query_bbox, radar_query_bbox_delta)


        query_feat = self.norm3(self.ffn(query_feat))
        cls_score = self.cls_branch(query_feat)  # [B, Q, num_classes]
        bbox_pred = self.reg_branch(query_feat)  # [B, Q, code_size]
        bbox_pred = self.refine_bbox(query_bbox, bbox_pred)

        time_diff = img_metas[0]['time_diff']  # [B, F]
        if time_diff.shape[1] > 1:
            # print("--------------------------------time_diff--------------------------------", time_diff)
            time_diff = time_diff.clone()
            frame_intervals = time_diff[:, 1:] - time_diff[:, :-1]   # shape (8,3)
            avg_time_diff = frame_intervals.mean(dim=1, keepdim=True).unsqueeze(-1)  # shape (8,1,1)
            avg_time_diff[avg_time_diff < 1e-5] = 1.0
            bbox_pred[..., 8:] = bbox_pred[..., 8:] / avg_time_diff



        if DUMP.enabled:
            if get_dist_info()[0] == 0:
                query_bbox_dec = decode_bbox(query_bbox, self.pc_range)
                bbox_pred_dec = decode_bbox(bbox_pred, self.pc_range)
                cls_score_sig = torch.sigmoid(cls_score)
                torch.save(query_bbox_dec.cpu(), '{}/query_bbox_stage{}.pth'.format(DUMP.file_path, DUMP.stage_count))
                torch.save(bbox_pred_dec.cpu(), '{}/bbox_pred_stage{}.pth'.format(DUMP.file_path, DUMP.stage_count))
                torch.save(cls_score_sig.cpu(), '{}/cls_score_stage{}.pth'.format(DUMP.file_path, DUMP.stage_count))
                radar_query_bbox_dec = decode_bbox(radar_query_bbox, self.pc_range)
                torch.save(radar_query_bbox_dec.cpu(), '{}/radar_query_bbox_stage{}.pth'.format(DUMP.file_path, DUMP.stage_count))
                

        return query_feat, cls_score, bbox_pred


class SparseBEVSelfAttention(BaseModule):
    """Scale-adaptive Self Attention"""
    def __init__(self, embed_dims=256, num_heads=8, dropout=0.1, pc_range=[], init_cfg=None):
        super().__init__(init_cfg)
        self.pc_range = pc_range

        self.attention = MultiheadAttention(embed_dims, num_heads, dropout, batch_first=True)
        self.gen_tau = nn.Linear(embed_dims, num_heads)

    @torch.no_grad()
    def init_weights(self):
        nn.init.zeros_(self.gen_tau.weight)
        nn.init.uniform_(self.gen_tau.bias, 0.0, 2.0)

    def inner_forward(self, query_bbox, query_feat, pre_attn_mask):
        """
        query_bbox: [B, Q, 10]
        query_feat: [B, Q, C]
        """
        dist = self.calc_bbox_dists(query_bbox)
        tau = self.gen_tau(query_feat)  # [B, Q, 8]

        if DUMP.enabled:
            if get_dist_info()[0] == 0:
                torch.save(tau.cpu(), '{}/sasa_tau_stage{}.pth'.format(DUMP.file_path, DUMP.stage_count))

        tau = tau.permute(0, 2, 1)  # [B, 8, Q]
        attn_mask = dist[:, None, :, :] * tau[..., None]  # [B, 8, Q, Q]

        if pre_attn_mask is not None:  # for query denoising
            attn_mask[:, :, pre_attn_mask] = float('-inf')

        attn_mask = attn_mask.flatten(0, 1)  # [Bx8, Q, Q]
        return self.attention(query_feat, attn_mask=attn_mask)

    def forward(self, query_bbox, query_feat, pre_attn_mask):
        if self.training and query_feat.requires_grad and getattr(self, 'checkpoint_enabled', True):
            return cp(self.inner_forward, query_bbox, query_feat, pre_attn_mask, use_reentrant=False)
        else:
            return self.inner_forward(query_bbox, query_feat, pre_attn_mask)

    @torch.no_grad()
    def calc_bbox_dists(self, bboxes):
        centers = decode_bbox(bboxes, self.pc_range)[..., :2]  # [B, Q, 2]

        if getattr(self, 'vectorized_distances', False):
            return -torch.norm(centers[:, :, None, :] - centers[:, None, :, :], dim=-1)

        dist = []
        for b in range(centers.shape[0]):
            dist_b = torch.norm(centers[b].reshape(-1, 1, 2) - centers[b].reshape(1, -1, 2), dim=-1)
            dist.append(dist_b[None, ...])

        dist = torch.cat(dist, dim=0)  # [B, Q, Q]
        dist = -dist

        return dist


class SparseBEVSampling(BaseModule):
    """Adaptive Spatio-temporal Sampling"""
    def __init__(self, embed_dims=256, num_frames=4, num_groups=4, num_points=8, num_levels=4, pc_range=[], num_cams=1, init_cfg=None):
        super().__init__(init_cfg)

        self.num_frames = num_frames
        self.num_points = num_points
        self.num_groups = num_groups
        self.num_levels = num_levels
        self.pc_range = pc_range

        self.sampling_offset = nn.Linear(embed_dims, num_groups * num_points * 3)
        # print(num_groups, num_points, num_levels)
        self.scale_weights = nn.Linear(embed_dims, num_groups * num_points * num_levels)
        self.num_cams = num_cams

    def init_weights(self):
        bias = self.sampling_offset.bias.data.view(self.num_groups * self.num_points, 3)
        nn.init.zeros_(self.sampling_offset.weight)
        nn.init.uniform_(bias[:, 0:3], -0.5, 0.5)

    def inner_forward(self, query_bbox, query_feat, mlvl_feats, img_metas):
        '''
        query_bbox: [B, Q, 10]
        query_feat: [B, Q, C]
        '''
        B, Q = query_bbox.shape[:2]
        image_h, image_w, _ = img_metas[0]['img_shape'][0]

        # sampling offset of all frames
        sampling_offset = self.sampling_offset(query_feat)
        sampling_offset = sampling_offset.view(B, Q, self.num_groups * self.num_points, 3)
        sampling_points = make_sample_points(query_bbox, sampling_offset, self.pc_range)  # [B, Q, GP, 3]
        sampling_points = sampling_points.reshape(B, Q, 1, self.num_groups, self.num_points, 3)
        sampling_points = sampling_points.expand(B, Q, self.num_frames, self.num_groups, self.num_points, 3)

        # warp sample points based on velocity
        time_diff = img_metas[0]['time_diff']  # [B, F]
        time_diff = time_diff[:, None, :, None]  # [B, 1, F, 1]
        vel = query_bbox[..., 8:].detach()  # [B, Q, 2]
        vel = vel[:, :, None, :]  # [B, Q, 1, 2]
        dist = vel * time_diff  # [B, Q, F, 2]
        dist = dist[:, :, :, None, None, :]  # [B, Q, F, 1, 1, 2]
        sampling_points = torch.cat([
            sampling_points[..., 0:2] - dist,
            sampling_points[..., 2:3]
        ], dim=-1)

        # scale weights
        scale_weights = self.scale_weights(query_feat).view(B, Q, self.num_groups, 1, self.num_points, self.num_levels)
        scale_weights = torch.softmax(scale_weights, dim=-1)
        scale_weights = scale_weights.expand(B, Q, self.num_groups, self.num_frames, self.num_points, self.num_levels)

        # sampling
        sampled_feats = sampling_4d(
            sampling_points,
            mlvl_feats,
            scale_weights,
            img_metas[0]['lidar2img'],
            image_h, image_w,
            self.num_cams
        )  # [B, Q, G, FP, C]

        return sampled_feats

    def forward(self, query_bbox, query_feat, mlvl_feats, img_metas):
        if self.training and query_feat.requires_grad and getattr(self, 'checkpoint_enabled', True):
            return cp(self.inner_forward, query_bbox, query_feat, mlvl_feats, img_metas, use_reentrant=False)
        else:
            return self.inner_forward(query_bbox, query_feat, mlvl_feats, img_metas)


class AdaptiveMixing(nn.Module):
    """Adaptive Mixing"""
    def __init__(self, in_dim, in_points, n_groups=1, query_dim=None, out_dim=None, out_points=None):
        super(AdaptiveMixing, self).__init__()

        out_dim = out_dim if out_dim is not None else in_dim
        out_points = out_points if out_points is not None else in_points
        query_dim = query_dim if query_dim is not None else in_dim

        self.query_dim = query_dim
        self.in_dim = in_dim
        self.in_points = in_points
        self.n_groups = n_groups
        self.out_dim = out_dim
        self.out_points = out_points

        self.eff_in_dim = in_dim // n_groups
        self.eff_out_dim = out_dim // n_groups

        self.m_parameters = self.eff_in_dim * self.eff_out_dim
        self.s_parameters = self.in_points * self.out_points
        self.total_parameters = self.m_parameters + self.s_parameters

        self.parameter_generator = nn.Linear(self.query_dim, self.n_groups * self.total_parameters)
        self.out_proj = nn.Linear(self.eff_out_dim * self.out_points * self.n_groups, self.query_dim)
        self.act = nn.ReLU(inplace=True)

    @torch.no_grad()
    def init_weights(self):
        nn.init.zeros_(self.parameter_generator.weight)

    def inner_forward(self, x, query):
        B, Q, G, P, C = x.shape
        assert G == self.n_groups
        assert P == self.in_points
        assert C == self.eff_in_dim

        '''generate mixing parameters'''
        params = self.parameter_generator(query)
        params = params.reshape(B*Q, G, -1)
        out = x.reshape(B*Q, G, P, C)

        M, S = params.split([self.m_parameters, self.s_parameters], 2)
        M = M.reshape(B*Q, G, self.eff_in_dim, self.eff_out_dim)
        S = S.reshape(B*Q, G, self.out_points, self.in_points)

        '''adaptive channel mixing'''
        out = torch.matmul(out, M)
        out = F.layer_norm(out, [out.size(-2), out.size(-1)])
        out = self.act(out)

        '''adaptive point mixing'''
        out = torch.matmul(S, out)  # implicitly transpose and matmul
        out = F.layer_norm(out, [out.size(-2), out.size(-1)])
        out = self.act(out)

        '''linear transfomation to query dim'''
        out = out.reshape(B, Q, -1)
        out = self.out_proj(out)
        out = query + out

        return out

    def forward(self, x, query):
        if self.training and x.requires_grad and getattr(self, 'checkpoint_enabled', True):
            return cp(self.inner_forward, x, query, use_reentrant=False)
        else:
            return self.inner_forward(x, query)
