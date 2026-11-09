
from typing import Any, Literal
import os.path as osp
import os
try:
    import torch
except ImportError:
    torch = None


seed = 42  # ⚠️ 训练可重复性：已设置随机种子，但需确保所有随机源都被固定
dataset_type = 'CustomNuScenesDataset'
dataset_root = os.environ.get('STREAMRC_DATA_ROOT', 'data/nuscenes/')
nusc_version = 'full'
output_dir = 'r50_fusion_streaming_full'

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=True,
    use_map=False,
    use_external=False
)

class_names = [
    'uav.drone'
]

point_cloud_range = [-25.6, -25.6, 0.0, 25.6, 25.6, 10]
voxel_size = [0.2, 0.2, 10]
post_center_range=[-30, -30, -5.0, 30, 30, 15.0] #nms阶段粗过滤
grid_size = [256, 256, 1]        # 可按实际需求取整
code_weight=[2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
radar_point_num=64
embed_dims = 256
# Paper setting (SQRCF-UAV, 2 refinement stages, 88.44 mAP in the paper).
num_layers = 2
num_query = 900
num_frames = 1
num_levels = 4
num_points = 8

# 获取 GPU 数量
gpu_num = int(os.environ.get('WORLD_SIZE', '1'))
if 'WORLD_SIZE' not in os.environ and torch is not None and torch.cuda.is_available():
    gpu_num = torch.cuda.device_count()
    print(f"Detected GPU count: {gpu_num}")
total_epochs = 36
per_gpu_batch_size = 8
batch_size = per_gpu_batch_size * gpu_num
sample_per_epoch = int(10700/batch_size)
workers_per_gpu = 4

use_streaming = True  # 是否启用流式训练（启用后会使用QueryStack时序建模和GroupSampler）
sequences_split_num='target_len'  # ⚠️ 训练可重复性：动态分割可能导致不同训练中sequence边界不同
individual_init = False
use_query_stack = True
last_init_query_bbox = False

keep_consistent_seq_aug = True
img_color_aug=True
train_dataloader_shuffle=True

lr = 2e-4
update_stage = []
fusion_stage = [1,2,3,4,5]

ida_aug_conf = {
    'resize_lim': (0.45, 0.55),
    'final_dim': (320, 832),
    'bot_pct_lim': (0.0, 0.0),
    'rot_lim': (0.0, 0.0),
    'H': 1080, 'W': 1920,
    'rand_flip': False,
}

img_backbone = dict(
    type='ResNet',
    depth=50,
    num_stages=4,
    out_indices=(0, 1, 2, 3),
    frozen_stages=1,
    norm_cfg=dict(type='BN2d', requires_grad=True),
    norm_eval=True,
    style='pytorch',
    with_cp=False)

img_neck = dict(
    type='FPN',
    in_channels=[256, 512, 1024, 2048],
    out_channels=embed_dims,
    num_outs=num_levels)
img_norm_cfg = dict(
    mean=[123.675, 116.280, 103.530],
    std=[58.395, 57.120, 57.375],
    to_rgb=True)


radar_backbone = dict(
    type='PointWiseMLP',
    in_dim=6,
    embed_dim=256,
    hidden_dims=(64, 128)
)

model = dict(
    type='SparseFusion_radarquery',
    data_aug=dict(
        img_color_aug=img_color_aug,  # GPU数据增强
        aug_seed=seed,  # ✅ 训练可重复性：设置GPU数据增强的随机种子，确保每次训练增强结果一致
        keep_consistent_seq_aug=keep_consistent_seq_aug,  # 可选：同一sequence内的样本使用相同的数据增强参数（流式训练时启用）
        img_norm_cfg=img_norm_cfg,
        img_pad_cfg=dict(size_divisor=32)),
    stop_prev_grad=0,
    img_backbone=img_backbone,
    img_neck=img_neck,
    pts_backbone=radar_backbone,

    pts_bbox_head=dict(
        type='SparseBEVHead_radarquery',
        num_classes=1,
        in_channels=embed_dims,
        num_query=num_query,
        query_denoising=True,
        query_denoising_groups=10,
        code_size=10,
        code_weights=code_weight,
        individual_init=individual_init,
        use_query_stack=use_query_stack,
        last_init_query_bbox=last_init_query_bbox,
        update_stage=update_stage,
        fusion_stage=fusion_stage,
        sync_cls_avg_factor=True,
        transformer=dict(
            type='SparseBEVTransformer_radarquery',
            embed_dims=embed_dims,
            num_frames=num_frames,
            num_points=num_points,
            num_layers=num_layers,
            num_levels=num_levels,
            num_classes=1,
            code_size=10,
            pc_range=point_cloud_range,
            num_cams=1,
            ),  # 流式训练：启用QueryStack时序建模
        bbox_coder=dict(
            type='NMSFreeCoder',
            post_center_range=post_center_range,
            pc_range=point_cloud_range,
            max_num=300,
            voxel_size=voxel_size,
            score_threshold=0.05,
            num_classes=1),
        positional_encoding=dict(
            type='SinePositionalEncoding',
            num_feats=embed_dims // 2,
            normalize=True,
            offset=-0.5),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0)),
    train_cfg=dict(pts=dict(
        grid_size=grid_size,
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        out_size_factor=4,
        assigner=dict(
            type='HungarianAssigner3D',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
            iou_cost=dict(type='IoUCost', weight=0.0),
        )
    ))
)


train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=False, color_type='color'),
    dict(type='SampleRadarPoints', num_points=radar_point_num, pts_dim=6),  # ⚠️ 训练可重复性：雷达点采样可能存在随机性，需确保使用固定种子
    dict(type='LoadMultiViewImageFromMultiSweeps', sweeps_num=num_frames - 1),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='RandomTransformImage', ida_aug_conf=ida_aug_conf, training=False),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['gt_bboxes_3d', 'gt_labels_3d', 'img', 'points'], meta_keys=(
        'filename', 'ori_shape', 'img_shape', 'pad_shape', 'lidar2img', 'img_timestamp', 'valid_height',
        'sequence_token', 'sequence_idx', 'sample_idx')  # 流式训练：添加sequence相关信息
    )
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=False, color_type='color'),
    dict(type='SampleRadarPoints', num_points=radar_point_num, pts_dim=6),
    dict(type='LoadMultiViewImageFromMultiSweeps', sweeps_num=num_frames - 1, test_mode=True),
    dict(type='RandomTransformImage', ida_aug_conf=ida_aug_conf, training=False),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1920, 1080),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='DefaultFormatBundle3D', class_names=class_names, with_label=False),
            dict(type='Collect3D', keys=['img', 'points'], meta_keys=(
                'filename', 'box_type_3d', 'ori_shape', 'img_shape', 'pad_shape',
                'lidar2img', 'img_timestamp', 'valid_height',
                'sequence_token', 'sequence_idx', 'sample_idx'))  # 流式训练：添加sequence相关信息
        ])
]

data = dict(
    workers_per_gpu=workers_per_gpu,  # ⚠️ 训练可重复性：多进程数据加载需要确保每个worker正确设置随机种子（seed + worker_id）
    train=dict(
        type=dataset_type,
        data_root=dataset_root,
        ann_file=osp.join(dataset_root, nusc_version, 'streamrc', 'nuscenes_infos_train_mingfusion_sweep.pkl'),
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        # ========== 流式训练配置 ==========
        sequences_split_num=sequences_split_num,  # sequence分割数量，1表示不分割（流式训练时启用，非流式时也设为1避免None）
        with_seq_flag=use_streaming,  # 启用sequence group flag（流式训练必需）
        keep_consistent_seq_aug=keep_consistent_seq_aug,  # 同一sequence内保持数据增强一致性（流式训练时启用）
        box_type_3d='LiDAR'),
    val=dict(
        type=dataset_type,
        data_root=dataset_root,
        ann_file=osp.join(dataset_root, nusc_version, 'streamrc', 'nuscenes_infos_val_mingfusion_sweep.pkl'),
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        sequences_split_num=1,
        with_seq_flag=True,
        box_type_3d='LiDAR'),
    test=dict(
        type=dataset_type,
        data_root=dataset_root,
        ann_file=osp.join(dataset_root, nusc_version, 'streamrc', 'nuscenes_infos_val_mingfusion_sweep.pkl'),
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        sequences_split_num=sequences_split_num,
        with_seq_flag=use_streaming,  # 测试时也启用sequence flag（流式训练时启用）
        box_type_3d='LiDAR'),
)


optimizer = dict(
    type='AdamW',
    lr=lr,
    paramwise_cfg=dict(custom_keys={
        'img_backbone': dict(lr_mult=0.1),
        'sampling_offset': dict(lr_mult=0.1),
    }),
    weight_decay=0.01
)
# fp16 = dict(loss_scale='dynamic')
optimizer_config = dict(
    type='Fp16OptimizerHook',  # ⚠️ 训练可重复性：FP16混合精度训练存在固有数值不确定性，可能导致2-3%性能差异
    loss_scale=512.0,  # 固定loss_scale，但FP16的舍入误差累积仍可能导致差异
    grad_clip=dict(max_norm=35, norm_type=2),
)
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=sample_per_epoch,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3
)

# load pretrained weights
# Set STREAMRC_LOAD_FROM=none to train without the nuImages-pretrained R50 weights.
load_from = os.environ.get('STREAMRC_LOAD_FROM', 'pretrain/cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim_20201009_124951-40963960.pth')
if load_from.lower() in ('', 'none'):
    load_from = None
revise_keys = [('backbone', 'img_backbone')]
resume_from = None
checkpoint_config = dict(interval=2, max_keep_ckpts=5)
eval_config = dict(interval=2)
debug = False
log_config = dict(
    interval=1,
    hooks=[
        dict(type='MyTextLoggerHook', interval=1, reset_flag=True),
        dict(type='MyTensorboardLoggerHook', interval=sample_per_epoch, reset_flag=True),
        dict(
            type='TensorboardLoggerHook',
            by_epoch=True,   # 按 iteration 记录
        ),
    ]
)
