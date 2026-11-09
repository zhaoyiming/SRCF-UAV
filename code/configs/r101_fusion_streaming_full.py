_base_ = ['./r50_fusion_streaming_full.py']
import os
import os.path as osp

# For nuScenes we usually do 10-class detection
class_names = [
    'uav.drone'
]

output_dir = 'r101_fusion_streaming_full'
per_gpu_batch_size = int(os.environ.get('STREAMRC_BATCH_SIZE_PER_GPU', '8'))
batch_size = per_gpu_batch_size * int(os.environ.get('WORLD_SIZE', '1'))
lr_warmup_epochs = 1

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-25.6, -25.6, 0.0, 25.6, 25.6, 10]
voxel_size = [0.2, 0.2, 10]
post_center_range=[-30, -30, -5.0, 30, 30, 15.0] #nms阶段粗过滤
grid_size = [256, 256, 1]        # 可按实际需求取整
code_weight=[2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

# num_sweeps=4

sample_per_epoch = int(10700/16/2)


img_backbone = dict(
    type='ResNet',
    depth=101,
    num_stages=4,
    out_indices=(0, 1, 2, 3),
    frozen_stages=1,
    norm_cfg=dict(type='BN2d', requires_grad=True),
    norm_eval=True,
    style='pytorch',
    with_cp=False,
    pretrained=os.environ.get('STREAMRC_R101_PRETRAINED', 'pretrain/resnet101-63fe2227.pth'),
)

img_neck = dict(
    type='FPN',
    in_channels=[256, 512, 1024, 2048],
    out_channels=256,
    num_outs=5,
)

model = dict(
    img_backbone=img_backbone,
    img_neck=img_neck,
    pts_bbox_head=dict(transformer=dict(num_levels=5)),
)

ida_aug_conf = {
    'resize_lim': (0.45 * 2, 0.55 * 2),
    'final_dim': (640, 1664),
    'bot_pct_lim': (0.0, 0.0),
    'rot_lim': (0.0, 0.0),
    'H': 1080, 'W': 1920,
    'rand_flip': False,
}




optimizer = dict(
    type='AdamW',
    lr=1e-4,
    paramwise_cfg=dict(custom_keys={
        'img_backbone': dict(lr_mult=0.1),
        'sampling_offset': dict(lr_mult=0.1),
    }),
    weight_decay=0.05
)


data = dict(
    workers_per_gpu=4,
)

# 覆盖基础配置中的 load_from（避免加载 ResNet50 权重到 ResNet101）
load_from = None
revise_keys = None
