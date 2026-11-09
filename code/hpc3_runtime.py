"""Use MMDetection's equivalent Torch focal loss if MMCV lacks a CUDA kernel."""
import logging

import torch
import torch.nn.functional as F
from mmdet.models.builder import LOSSES
from mmdet.models.losses.focal_loss import FocalLoss, py_sigmoid_focal_loss


@LOSSES.register_module()
class TorchFocalLoss(FocalLoss):
    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None):
        if self.activated:
            return super().forward(pred, target, weight, avg_factor, reduction_override)
        assert reduction_override in (None, 'none', 'mean', 'sum')
        num_classes = pred.size(1)
        target = F.one_hot(target, num_classes=num_classes + 1)[:, :num_classes]
        return self.loss_weight * py_sigmoid_focal_loss(
            pred, target, weight, gamma=self.gamma, alpha=self.alpha,
            reduction=reduction_override or self.reduction, avg_factor=avg_factor)


def configure_focal_loss(cfg):
    loss = cfg.model.pts_bbox_head.loss_cls
    if loss.type != 'FocalLoss':
        return loss.type
    try:
        pred = torch.zeros(2, 1, device='cuda', requires_grad=True)
        target = torch.tensor([0, 1], device='cuda')
        FocalLoss()(pred, target).backward()
        torch.cuda.synchronize()
    except RuntimeError as exc:
        message = str(exc).lower()
        if not any(s in message for s in ['implementation for device', 'not compiled with gpu', 'not implemented for']):
            raise
        loss.type = 'TorchFocalLoss'
        logging.warning('MMCV CUDA focal loss unavailable; using MMDetection Torch formula: %s', exc)
    return loss.type
