"""Stop corrupted training instead of letting sanitized predictions hide it."""
import json
import math
from pathlib import Path

from mmcv.runner import Hook


class TrainingHealthHook(Hook):
    def __init__(self, warmup=50, loss_multiplier=10., minimum_limit=100.):
        self.warmup = warmup
        self.loss_multiplier = loss_multiplier
        self.minimum_limit = minimum_limit
        self.seen = 0
        self.loss_ema = None

    def after_train_iter(self, runner):
        loss = float(runner.outputs['loss'].detach())
        limit = max(self.minimum_limit, self.loss_multiplier * (self.loss_ema or 0.))
        if not math.isfinite(loss) or (self.seen >= self.warmup and loss > limit):
            report = dict(epoch=runner.epoch + 1, iteration=runner.iter + 1,
                          loss=loss, previous_loss_ema=self.loss_ema, limit=limit)
            if runner.rank == 0:
                Path(runner.work_dir, 'training_health_failure.json').write_text(
                    json.dumps(report, indent=2) + '\n')
            raise FloatingPointError('Training health check failed: ' + str(report))
        self.loss_ema = loss if self.loss_ema is None else 0.98 * self.loss_ema + 0.02 * loss
        self.seen += 1
