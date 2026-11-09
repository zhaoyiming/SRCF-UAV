"""A bounded benchmark of the real MMCV training loop, including data loading."""
import functools
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from mmcv.runner import Hook


class BenchmarkComplete(Exception):
    """Stop without entering validation or saving an epoch checkpoint."""


class ThroughputHook(Hook):
    def __init__(self, warmup, iterations, output, metadata):
        self.warmup = warmup
        self.iterations = iterations
        self.output = output
        self.metadata = metadata
        self.seen = 0
        self.updates = 0
        self.frames = 0
        self.losses = []
        self.input_shape = None
        self.start = None

    def before_run(self, runner):
        self.original_step = runner.optimizer.step

        @functools.wraps(self.original_step)
        def counted_step(*args, **kwargs):
            result = self.original_step(*args, **kwargs)
            self.updates += 1
            return result

        runner.optimizer.step = counted_step
        module = runner.model.module
        self.parameter_name, self.parameter = next(
            (name, p) for name, p in module.named_parameters() if p.requires_grad)
        self.initial_parameter = self.parameter.detach().float().clone()

        def capture_shape(module, inputs):
            if self.input_shape is None:
                self.input_shape = list(inputs[0].shape)

        self.shape_handle = module.img_backbone.register_forward_pre_hook(capture_shape)

    def _actual_updates(self, runner):
        state = runner.optimizer.state.get(self.parameter, {})
        if runner.optimizer.defaults.get('fused') and 'step' in state:
            return int(state['step'].item())
        return self.updates

    def _start(self, runner):
        torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        self.warmup_updates = self._actual_updates(runner)
        self.start = time.perf_counter()

    def before_train_epoch(self, runner):
        if len(runner.data_loader) < self.warmup + self.iterations:
            raise ValueError('Benchmark must fit in one training epoch.')
        if self.warmup == 0:
            self._start(runner)

    def after_train_iter(self, runner):
        self.seen += 1
        loss = float(runner.outputs['loss'].detach())
        if not math.isfinite(loss):
            raise RuntimeError('Non-finite training loss during benchmark.')
        if self.seen <= self.warmup:
            if self.seen == self.warmup:
                self._start(runner)
                runner.logger.info('Benchmark warmup complete; timing %d iterations.', self.iterations)
            return
        self.frames += int(runner.outputs['num_samples'])
        self.losses.append(loss)
        if self.seen == self.warmup + self.iterations:
            self._finish(runner)
            raise BenchmarkComplete()

    def _finish(self, runner):
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - self.start
        rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        # Audit outside the timed region: the last optimizer update must not
        # corrupt weights or Adam moments even if the preceding loss was finite.
        tensors = list(runner.model.parameters())
        tensors += [value for state in runner.optimizer.state.values()
                    for value in state.values() if torch.is_tensor(value)]
        finite = bool(torch.stack([torch.isfinite(t.detach()).all().to(self.parameter.device)
                                   for t in tensors]).all())
        local = dict(
            rank=rank, gpu=torch.cuda.get_device_name(), seconds=elapsed,
            model_and_optimizer_finite=finite,
            frames=self.frames, measured_optimizer_updates=self._actual_updates(runner) - self.warmup_updates,
            total_optimizer_updates=self._actual_updates(runner),
            peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
            device_total_gib=torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / 2**30,
            loss_first=self.losses[0], loss_last=self.losses[-1],
            loss_min=min(self.losses), loss_max=max(self.losses),
            image_backbone_input_shape=self.input_shape,
            parameter_checked=self.parameter_name,
            parameter_max_change=float((self.parameter.detach().float() - self.initial_parameter).abs().max()),
        )
        ranks = [None] * world_size
        if dist.is_initialized():
            dist.all_gather_object(ranks, local)
        else:
            ranks[0] = local
        seconds = max(item['seconds'] for item in ranks)
        frames = sum(item['frames'] for item in ranks)
        valid = all(item['measured_optimizer_updates'] == self.iterations
                    and item['parameter_max_change'] > 0
                    and item['model_and_optimizer_finite'] for item in ranks)
        report = dict(
            **self.metadata, world_size=world_size, warmup_iterations=self.warmup,
            measured_iterations=self.iterations, elapsed_seconds=seconds,
            global_frames=frames, global_frames_per_second=frames / seconds,
            iterations_per_second=self.iterations / seconds,
            seconds_per_iteration=seconds / self.iterations,
            valid_optimizer_updates=valid,
            scope='real training: data loading, augmentation, forward, losses, backward, optimizer, DDP, iteration logging; no eval/checkpoint',
            ranks=ranks,
        )
        if rank == 0:
            path = Path(self.output)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix('.tmp')
            temporary.write_text(json.dumps(report, indent=2) + '\n')
            os.replace(str(temporary), str(path))
            runner.logger.info('THROUGHPUT_RESULT %s', json.dumps(report))
        if not valid:
            raise RuntimeError('Optimizer updates were skipped or the checked parameter did not change; see benchmark JSON.')

    def after_run(self, runner):
        runner.optimizer.step = self.original_step
        self.shape_handle.remove()
