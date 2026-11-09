"""Measure complete training epochs separately from validation/checkpoint time."""
import functools
import json
import math
import time
from pathlib import Path

import torch
import torch.distributed as dist
from mmcv.runner import Hook


class TrainingLimitReached(Exception):
    """Gracefully finish a bounded run without changing its full LR schedule."""


def gather(value):
    if not dist.is_initialized():
        return [value]
    values = [None] * dist.get_world_size()
    dist.all_gather_object(values, value)
    return values


def sync():
    torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()
    torch.cuda.synchronize()


class EpochTimingHook(Hook):
    def __init__(self, metadata):
        self.metadata = metadata
        self.updates = 0
        self.input_shape = None

    def before_run(self, runner):
        self.original_step = runner.optimizer.step

        @functools.wraps(self.original_step)
        def counted_step(*args, **kwargs):
            result = self.original_step(*args, **kwargs)
            self.updates += 1
            return result

        runner.optimizer.step = counted_step

        def capture_shape(module, inputs):
            if self.input_shape is None:
                self.input_shape = list(inputs[0].shape)

        self.shape_handle = runner.model.module.img_backbone.register_forward_pre_hook(capture_shape)
        self.tracked_parameter = next(p for p in runner.model.module.parameters() if p.requires_grad)

    def _actual_updates(self, runner):
        state = runner.optimizer.state.get(self.tracked_parameter, {})
        if runner.optimizer.defaults.get('fused'):
            return int(state['step'].item()) if 'step' in state else 0
        return self.updates

    def before_train_epoch(self, runner):
        sync()
        torch.cuda.reset_peak_memory_stats()
        self.start_updates = self._actual_updates(runner)
        self.frames = 0
        self.iterations = 0
        self.loss_first = None
        self.loss_last = None
        self.start = time.perf_counter()

    def after_train_iter(self, runner):
        loss = float(runner.outputs['loss'].detach())
        if not math.isfinite(loss):
            raise RuntimeError('Non-finite training loss.')
        if self.loss_first is None:
            self.loss_first = loss
        self.loss_last = loss
        self.frames += int(runner.outputs['num_samples'])
        self.iterations += 1

    def after_train_epoch(self, runner):
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - self.start
        local = dict(
            rank=dist.get_rank() if dist.is_initialized() else 0,
            train_seconds=elapsed, frames=self.frames, iterations=self.iterations,
            optimizer_updates=self._actual_updates(runner) - self.start_updates,
            image_backbone_input_shape=self.input_shape,
            peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
            device_total_gib=torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / 2**30,
            loss_first=self.loss_first, loss_last=self.loss_last,
        )
        ranks = gather(local)
        seconds = max(rank['train_seconds'] for rank in ranks)
        frames = sum(rank['frames'] for rank in ranks)
        self.report = dict(
            **self.metadata, epoch=runner.epoch + 1, world_size=len(ranks),
            train_seconds=seconds, global_frames=frames,
            global_frames_per_second=frames / seconds,
            all_optimizer_updates_succeeded=all(rank['optimizer_updates'] == rank['iterations'] for rank in ranks),
            ranks=ranks,
        )
        if runner.rank == 0:
            runner.logger.info('EPOCH_TRAIN_TIME epoch=%d steps=%d seconds=%.3f global_fps=%.3f peak_allocated_gib=%.3f',
                               runner.epoch + 1, self.iterations, seconds, frames / seconds,
                               max(rank['peak_allocated_gib'] for rank in ranks))

    def after_run(self, runner):
        runner.optimizer.step = self.original_step
        self.shape_handle.remove()


class EpochEndTimingHook(Hook):
    def __init__(self, timing, stop_after_epochs=None):
        self.timing = timing
        self.stop_after_epochs = stop_after_epochs

    def after_train_epoch(self, runner):
        sync()
        elapsed = time.perf_counter() - self.timing.start
        totals = gather(elapsed)
        report = dict(self.timing.report)
        report['total_epoch_seconds'] = max(totals)
        report['validation_checkpoint_logging_seconds'] = report['total_epoch_seconds'] - report['train_seconds']
        if runner.rank == 0:
            path = Path(runner.work_dir) / 'epoch_times.jsonl'
            with path.open('a') as handle:
                handle.write(json.dumps(report) + '\n')
            runner.logger.info('EPOCH_TOTAL_TIME %s', json.dumps(report))
        if self.stop_after_epochs is not None and runner.epoch + 1 >= self.stop_after_epochs:
            raise TrainingLimitReached()
