# -*- coding: utf-8 -*-
"""
Fast distributed evaluation hook with rank-ordered concatenation.

Compatibility:
- Accepts `gpu_collect` in __init__ for drop-in replacement.

Key features:
1) In-memory gather via pure dist.all_gather (tensor-based, official-style).
2) Optional disk fallback (pickle parts into tmpdir) when you don't want gpu/tensor gather.
3) Override evaluate() to pass eval_kwargs as **kwargs (compat with custom dataset.evaluate signature).
4) Barrier after evaluation to avoid DDP hang.

IMPORTANT:
- Rank-ordered concatenation (use_zip_sort=False) is correct ONLY if your validation sampler assigns
  each rank a contiguous chunk of dataset indices (contiguous-split).
- If your sampler is stride-based (e.g., DistributedSampler), set use_zip_sort=True to use zip-based
  reordering (same as mmdet's multi_gpu_test).
"""

import bisect
import json
import os
import os.path as osp
import pickle
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, List, Optional, Tuple

import torch
import torch.distributed as dist
from mmcv.runner import get_dist_info
from mmcv.utils import is_list_of
from mmdet.core import DistEvalHook as BaseDistEvalHook
from torch.nn.modules.batchnorm import _BatchNorm


def _calc_dynamic_intervals(start_interval, dynamic_interval_list):
    assert is_list_of(dynamic_interval_list, tuple)

    dynamic_milestones = [0]
    dynamic_milestones.extend(
        [dynamic_interval[0] for dynamic_interval in dynamic_interval_list])
    dynamic_intervals = [start_interval]
    dynamic_intervals.extend(
        [dynamic_interval[1] for dynamic_interval in dynamic_interval_list])
    return dynamic_milestones, dynamic_intervals


def _is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def _barrier():
    if _is_dist():
        dist.barrier()


def _mkdir(path: str):
    os.makedirs(path, exist_ok=True)


def _rank_world() -> Tuple[int, int]:
    return get_dist_info()


def _default_device() -> torch.device:
    # Match common mmdet/mmcv behavior: use cuda when available.
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def _to_uint8_tensor(obj: Any, device: torch.device) -> torch.Tensor:
    # Official-style: pickle -> bytearray -> uint8 tensor on device
    payload = bytearray(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))
    return torch.tensor(payload, dtype=torch.uint8, device=device)


# ---------------------------
# In-memory gather (OFFICIAL-STYLE dist.all_gather)
# ---------------------------

def collect_results_concat_in_memory(
    result_part: List[Any],
    size: int,
    debug: bool = False,
    use_zip_sort: bool = True,  # True: stride-based sampler, False: contiguous-split
) -> Optional[List[Any]]:
    rank, world_size = _rank_world()

    if (not _is_dist()) or world_size == 1:
        return result_part[:size]

    device = _default_device()

    # 1) serialize local results to uint8 tensor
    part_tensor = _to_uint8_tensor(result_part, device=device)

    # 2) gather tensor lengths
    shape_tensor = torch.tensor(part_tensor.shape, device=device)  # (N,) -> tensor([N])
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)

    # 3) pad to max length then all_gather
    shape_max = torch.stack(shape_list).max().item()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device=device)
    part_send[: shape_tensor[0].item()] = part_tensor

    part_recv_list = [part_send.new_zeros(shape_max) for _ in range(world_size)]
    dist.all_gather(part_recv_list, part_send)

    if rank != 0:
        return None

    # 4) deserialize each rank's bytes back to Python list
    part_list: List[List[Any]] = []
    for recv, shape in zip(part_recv_list, shape_list):
        n = int(shape[0].item())
        buf = recv[:n].cpu().numpy().tobytes()
        part = pickle.loads(buf)
        if part is None:
            part_list.append([])
        elif isinstance(part, list):
            part_list.append(part)
        else:
            part_list.append(list(part))

    # 5) reorder / concat
    if use_zip_sort:
        # stride-based sampler: interleave (official mmdet behavior)
        ordered: List[Any] = []
        for res in zip(*part_list):
            ordered.extend(list(res))
    else:
        # contiguous-split sampler: rank-ordered concatenation
        ordered = []
        for r in range(world_size):
            ordered.extend(part_list[r])

    # 6) trim to dataset size (dataloader may pad)
    ordered = ordered[:size]

    if debug and rank == 0:
        print(
            f"[collect_results_concat_in_memory] collected={len(ordered)} (size={size}), "
            f"use_zip_sort={use_zip_sort}, device={device}"
        )

    return ordered


# ---------------------------
# Disk fallback gather (keep official-style CPU/disk behavior; add concat mode)
# ---------------------------

def collect_results_cpu_concat_disk(
    result_part: List[Any],
    size: int,
    tmpdir: Optional[str],
    cleanup: bool = True,
    debug: bool = False,
    use_zip_sort: bool = True,  # True: stride-based sampler, False: contiguous-split
) -> Optional[List[Any]]:
    rank, world_size = _rank_world()

    # 1) create/broadcast tmpdir (official mmcv-style broadcast with ByteTensor)
    if tmpdir is None:
        if rank == 0:
            tmpdir = tempfile.mkdtemp(prefix="mmcv_eval_concat_")
        else:
            tmpdir = ""
        if _is_dist():
            max_len = 512
            raw = tmpdir.encode("utf-8")
            if len(raw) > max_len:
                raise RuntimeError(f"tmpdir too long: {tmpdir}")
            buf = raw + b"\x00" * (max_len - len(raw))
            t = torch.ByteTensor(list(buf))
            if torch.cuda.is_available():
                t = t.cuda()
            dist.broadcast(t, src=0)
            tmpdir = bytes(t.cpu().tolist()).split(b"\x00", 1)[0].decode("utf-8")

    _mkdir(tmpdir)

    # 2) dump per-rank part
    part_path = osp.join(tmpdir, f"part_{rank}.pkl")
    with open(part_path, "wb") as f:
        pickle.dump(result_part, f, protocol=pickle.HIGHEST_PROTOCOL)

    if debug:
        print(
            f"[collect_results_cpu_concat_disk] rank={rank} wrote {part_path} "
            f"({len(result_part)} items)"
        )

    _barrier()

    if rank != 0:
        return None

    # 3) load all parts on rank0
    part_list: List[List[Any]] = []
    for r in range(world_size):
        p = osp.join(tmpdir, f"part_{r}.pkl")
        if not osp.exists(p):
            raise FileNotFoundError(f"Missing part file: {p}")
        with open(p, "rb") as f:
            part = pickle.load(f)
        if part is None:
            part_list.append([])
        elif isinstance(part, list):
            part_list.append(part)
        else:
            part_list.append(list(part))

    # 4) reorder / concat (same policy as in-memory)
    if use_zip_sort:
        ordered: List[Any] = []
        for res in zip(*part_list):
            ordered.extend(list(res))
    else:
        ordered = []
        for r in range(world_size):
            ordered.extend(part_list[r])

    ordered = ordered[:size]

    if debug and rank == 0:
        print(
            f"[collect_results_cpu_concat_disk] rank0 collected={len(ordered)} (size={size}), "
            f"use_zip_sort={use_zip_sort}"
        )

    if cleanup:
        try:
            shutil.rmtree(tmpdir)
        except Exception as e:
            if debug:
                print(f"[collect_results_cpu_concat_disk] cleanup failed: {e}")

    return ordered


# ---------------------------
# Multi-GPU test
# ---------------------------

@torch.no_grad()
def multi_gpu_test_concat_fast(
    model,
    data_loader,
    tmpdir: Optional[str] = None,
    gpu_collect: bool = True,
    debug: bool = False,
    use_zip_sort: bool = False,  # True: stride-based sampler, False: contiguous-split
) -> Optional[List[Any]]:
    model.eval()
    if getattr(data_loader.batch_sampler, 'indexed_results', False):
        from streaming_val import parallel_streaming_test
        return parallel_streaming_test(model, data_loader, gpu_collect=gpu_collect)
    rank, world_size = _rank_world()
    dataset = data_loader.dataset

    results: List[Any] = []
    for i, data in enumerate(data_loader):
        out = model(return_loss=False, rescale=True, **data)

        if isinstance(out, (list, tuple)):
            results.extend(list(out))
        else:
            results.append(out)

        if debug and rank == 0 and (i % 50 == 0):
            print(f"[multi_gpu_test_concat_fast] progress {i}/{len(data_loader)}")

    if (not _is_dist()) or world_size == 1:
        return results[: len(dataset)]

    size = len(dataset)

    if gpu_collect:
        # tensor-based dist.all_gather (official-style)
        return collect_results_concat_in_memory(
            results, size=size, debug=debug, use_zip_sort=use_zip_sort
        )
    else:
        # disk fallback
        return collect_results_cpu_concat_disk(
            results,
            size=size,
            tmpdir=tmpdir,
            cleanup=True,
            debug=debug,
            use_zip_sort=use_zip_sort,
        )


class DistEvalHookConcatFast(BaseDistEvalHook):

    def __init__(self, *args, dynamic_intervals=None, batch_search=None, **kwargs):
        super(DistEvalHookConcatFast, self).__init__(*args, **kwargs)
        self.latest_results = None
        self.batch_search = batch_search
        self.batch_search_complete = False

        self.use_dynamic_intervals = dynamic_intervals is not None
        if self.use_dynamic_intervals:
            self.dynamic_milestones, self.dynamic_intervals = \
                _calc_dynamic_intervals(self.interval, dynamic_intervals)

    def _decide_interval(self, runner):
        if self.use_dynamic_intervals:
            progress = runner.epoch if self.by_epoch else runner.iter
            step = bisect.bisect(self.dynamic_milestones, (progress + 1))
            # Dynamically modify the evaluation interval
            self.interval = self.dynamic_intervals[step - 1]

    def before_train_epoch(self, runner):
        """Evaluate the model only at the start of training by epoch."""
        self._decide_interval(runner)
        super().before_train_epoch(runner)

    def before_train_iter(self, runner):
        self._decide_interval(runner)
        super().before_train_iter(runner)

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        if self.broadcast_bn_buffer:
            model = runner.model
            for name, module in model.named_modules():
                if isinstance(module, _BatchNorm) and module.track_running_stats:
                    dist.broadcast(module.running_var, 0)
                    dist.broadcast(module.running_mean, 0)

        if not self._should_evaluate(runner):
            return

        if self.batch_search and not self.batch_search_complete:
            from validation_tuning import select_validation_loader
            self.dataloader = select_validation_loader(
                runner, self.dataloader.dataset, self.batch_search)
            self.batch_search_complete = True

        tmpdir = self.tmpdir
        if tmpdir is None:
            tmpdir = osp.join(runner.work_dir, '.eval_hook')

        results = multi_gpu_test_concat_fast(
            runner.model,
            self.dataloader,
            tmpdir=tmpdir,
            gpu_collect=self.gpu_collect
        )
        self.latest_results = results

        if runner.rank == 0:
            print('\n')
            runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
            metric_start = time.perf_counter()
            key_score = self.evaluate(runner, results)
            metric_seconds = time.perf_counter() - metric_start

            save_start = time.perf_counter()
            if self.save_best and key_score:
                self._save_ckpt(runner, key_score)
            timings = getattr(runner.model, '_streaming_val_timings', None)
            if timings is not None:
                profile = dict(epoch=runner.epoch + 1, ranks=timings,
                               metrics_seconds=metric_seconds,
                               best_save_seconds=time.perf_counter() - save_start)
                runner.logger.info('VAL_PHASE_TIME %s', json.dumps(profile))
                with Path(runner.work_dir, 'val_phase_times.jsonl').open('a') as handle:
                    handle.write(json.dumps(profile) + '\n')
