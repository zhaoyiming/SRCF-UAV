"""Opt-in execution settings; model architecture and supervision stay in config."""
import functools
import importlib
import json
import logging
from types import SimpleNamespace

import torch


def configure_backbone_checkpointing(model):
    """Use non-reentrant ResNet checkpoints with DDP unused-parameter detection.

    Replace only the ResNet module's checkpoint API in this process. The
    installed framework and the decoder's checkpoint implementation stay intact.
    """
    source_modules = {type(module).__module__ for module in model.img_backbone.modules()
                      if getattr(module, 'with_cp', False)}
    changed = []
    from torch.utils.checkpoint import checkpoint
    for name in sorted(source_modules):
        source = importlib.import_module(name)
        if hasattr(source, 'cp'):
            source.cp = SimpleNamespace(checkpoint=functools.partial(
                checkpoint, use_reentrant=False))
            changed.append(name)
    return dict(implementation='non_reentrant' if changed else 'disabled', modules=changed)


def configure_compatible_sdpa():
    """PyTorch 2.0's fused attention backward kernels can fault on Hopper."""
    if (torch.cuda.is_available()
            and torch.cuda.get_device_capability()[0] >= 9
            and tuple(int(x) for x in torch.__version__.split('+')[0].split('.')[:2]) < (2, 1)):
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        return 'math (PyTorch 2.0 / Hopper compatibility)'
    return 'automatic'


def _discard_attention_weights(module):
    original = module.forward

    @functools.wraps(original)
    def forward(*args, **kwargs):
        # All callers in this detector discard the returned weight matrix.
        if len(args) > 4:
            args = list(args)
            args[4] = False
        else:
            kwargs['need_weights'] = False
        return original(*args, **kwargs)

    module.forward = forward


def configure_training_speed(model, settings):
    settings = dict(settings)
    backend = configure_compatible_sdpa()
    if not settings.get('enabled', False):
        return dict(enabled=False, sdpa_backend=backend)

    settings['backbone_checkpointing'] = configure_backbone_checkpointing(model)

    from models.query_stack import QueryStack
    from models.sparsebev_transformer_radarquery import (
        AdaptiveMixing, SparseBEVSampling, SparseBEVSelfAttention)

    torch.backends.cuda.matmul.allow_tf32 = settings.get('tf32', True)
    torch.backends.cudnn.allow_tf32 = settings.get('tf32', True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    attention_modules = 0
    for module in model.modules():
        if isinstance(module, QueryStack):
            module.skip_debug_statistics = True
        if isinstance(module, (AdaptiveMixing, SparseBEVSampling, SparseBEVSelfAttention)):
            module.checkpoint_enabled = settings.get('checkpointing', True)
        if isinstance(module, SparseBEVSelfAttention):
            module.vectorized_distances = settings.get('vectorized_distances', True)
        if isinstance(module, torch.nn.MultiheadAttention) and settings.get('sdpa', True):
            _discard_attention_weights(module)
            attention_modules += 1
    model.pts_bbox_head.single_gt_targets = settings.get('single_gt_targets', True)
    settings.update(attention_modules=attention_modules,
                    sdpa_backend=backend,
                    cudnn_benchmark=torch.backends.cudnn.benchmark,
                    cudnn_deterministic=torch.backends.cudnn.deterministic)
    logging.info('HPC3 execution settings: %s', settings)
    return settings


def configure_optimizer_speed(optimizer, settings):
    if not settings.get('enabled', False):
        return
    # MMCV creates one AdamW group per tensor when paramwise_cfg is present.
    # Coalesce identical hyperparameters so fused AdamW can process tensors together.
    groups = {}
    for group in optimizer.param_groups:
        key = json.dumps({k: v for k, v in group.items() if k != 'params'}, sort_keys=True)
        if key not in groups:
            groups[key] = dict(group, params=[])
        groups[key]['params'].extend(group['params'])
    previous = len(optimizer.param_groups)
    optimizer.param_groups = list(groups.values())
    logging.info('AdamW parameter groups: %d -> %d', previous, len(optimizer.param_groups))
    version = tuple(int(x) for x in torch.__version__.split('+')[0].split('.')[:2])
    if optimizer.defaults.get('fused') and version < (2, 1):
        # Torch 2.0 _fused_adamw drops found_inf when grad_scale is None.
        # MMCV explicitly unscales before clipping, so GradScaler passes exactly
        # that combination. Unit scale preserves already-unscaled gradients and
        # keeps the fused kernel's overflow skip active.
        original_step = optimizer.step
        device = optimizer.param_groups[0]['params'][0].device
        unit_scale = torch.ones((), dtype=torch.float32, device=device)

        @functools.wraps(original_step)
        def step(*args, **kwargs):
            if (getattr(optimizer, 'grad_scale', None) is None
                    and getattr(optimizer, 'found_inf', None) is not None):
                optimizer.grad_scale = unit_scale
            return original_step(*args, **kwargs)

        optimizer.step = step
        settings['amp_unscale_compat'] = True
        logging.info('Enabled fused AdamW overflow protection after explicit AMP unscale.')
