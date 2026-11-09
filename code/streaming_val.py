"""Batch independent streaming histories without changing their frame order."""
from collections import deque

import numpy as np
import torch
from torch.utils.data import Sampler


class ParallelStreamingValSampler(Sampler):
    """Keep the original rank split and batch independent temporal chains.

    QueryStack currently accepts history by timestamp, including across some
    dataset.flag boundaries. Split on that same rule, rather than dropping
    history at a flag boundary and silently changing the validation baseline.
    """

    indexed_results = True

    def __init__(self, dataset, samples_per_gpu, num_replicas=None, rank=None,
                 seed=0, max_time_interval=1.):
        from loaders.samplers.group_sampler import GroupInBatchValBatchSampler

        original = GroupInBatchValBatchSampler(
            dataset, 1, num_replicas, rank, seed=seed)
        self.indices = list(original.indices)
        self.batch_size = int(samples_per_gpu)
        if self.batch_size < 1:
            raise ValueError('Validation batch size must be positive')
        self.sampler = self
        self.epoch = 0
        chains = []
        previous = None
        for index in self.indices:
            camera = next(iter(dataset.data_infos[index]['cams'].values()))
            timestamp = camera['timestamp'] / 1e6
            # QueryStack casts the double-precision difference to FP32.
            delta = None if previous is None else float(np.float32(timestamp - previous))
            if delta is None or not 0. <= delta <= max_time_interval:
                chains.append([])
            chains[-1].append(index)
            previous = timestamp
        self.stream_for_index = {
            index: stream for stream, chain in enumerate(chains) for index in chain}
        self.final_indices = {chain[-1] for chain in chains}
        self.chain_lengths = [len(chain) for chain in chains]
        # Start long chains first so a long sequence does not occupy the tail
        # alone. Never split an actual causal dependency to fill a batch.
        pending = deque(sorted(chains, key=len, reverse=True)
                        if self.batch_size > 1 else chains)
        active = []
        self.batches = []
        while pending or active:
            while pending and len(active) < self.batch_size:
                active.append(deque(pending.popleft()))
            self.batches.append([chain.popleft() for chain in active])
            active = [chain for chain in active if chain]

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


def restore_result_order(indexed_results, size):
    ordered = sorted(indexed_results, key=lambda item: item[0])
    if [index for index, _ in ordered] != list(range(size)):
        raise RuntimeError('Validation results have missing or duplicate dataset indices')
    return [result for _, result in ordered]


def select_eval_rows(data, rows):
    """Select samples after MMCV scatter has unwrapped DataContainers."""
    if set(data) != {'img', 'img_metas', 'points'}:
        raise ValueError('Batched streaming validation expects img, img_metas and points')
    if len(data['img']) != 1 or len(data['img_metas']) != 1 or len(data['points']) != 1:
        raise ValueError('Batched streaming validation supports one test augmentation')
    return dict(
        img=[data['img'][0][rows]],
        img_metas=[[dict(data['img_metas'][0][i]) for i in rows]],
        points=[[data['points'][0][i] for i in rows]],
    )


class StreamingValCache:
    """Pack cached histories into a batch; run fresh frames separately.

    QueryStack resets its entire batch if one sample has no usable history.
    Separating fresh/history groups preserves the original batch-one model
    path without modifying model math or the training cache implementation.
    """

    def __init__(self, stack):
        self.stack = stack
        self.states = {}
        self.stack.reset()

    def predict(self, module, data, stream_ids):
        if len(set(stream_ids)) != len(stream_ids):
            raise ValueError('A validation batch contains dependent frames')
        outputs = [None] * len(stream_ids)
        cached_rows = {i for i, stream in enumerate(stream_ids) if stream in self.states}
        for has_history in (False, True):
            rows = [i for i, stream in enumerate(stream_ids)
                    if (i in cached_rows) == has_history]
            if not rows:
                continue
            subset = select_eval_rows(data, rows)
            self.stack.reset()
            if has_history:
                states = [self.states[stream_ids[i]] for i in rows]
                self.stack.historical_feature = torch.cat([x[0] for x in states])
                self.stack.historical_bbox = torch.cat([x[1] for x in states])
                self.stack.historical_metas = dict(
                    timestamp=torch.cat([x[2] for x in states]),
                    img_metas=[x[3] for x in states])
            # The current training uses this offline path on both ranks.
            # Calling it directly also permits the same check on one GPU.
            result = module.simple_test_offline(
                subset['img_metas'][0], subset['img'][0], subset['points'], rescale=True)
            if len(result) != len(rows):
                raise RuntimeError('Model output count does not match validation batch')
            if self.stack.historical_feature is None:
                raise RuntimeError('The model did not cache its streaming queries')
            for position, row in enumerate(rows):
                outputs[row] = result[position]
                self.states[stream_ids[row]] = (
                    self.stack.historical_feature[position:position + 1].detach().clone(),
                    self.stack.historical_bbox[position:position + 1].detach().clone(),
                    self.stack.historical_metas['timestamp'][position:position + 1].detach().clone(),
                    self.stack.historical_metas['img_metas'][position],
                )
        self.stack.reset()
        return outputs

    def close(self):
        self.states.clear()
        self.stack.reset()


@torch.no_grad()
def parallel_streaming_test(model, data_loader, gpu_collect=True):
    import time
    import torch.distributed as dist
    from mmcv.runner import get_dist_info
    from loaders.apis.test import (collect_results_concat_in_memory,
                                  collect_results_cpu_concat_disk)

    module = model.module
    sampler = data_loader.batch_sampler
    stack = module.pts_bbox_head.query_stack
    cache = StreamingValCache(stack)
    rank, world_size = get_dist_info()
    results = []
    torch.cuda.synchronize()
    start = time.perf_counter()
    try:
        for batch_number, data in enumerate(data_loader):
            indices = sampler.batches[batch_number]
            _, scattered = model.scatter((), data, [torch.cuda.current_device()])
            data = scattered[0]
            metas = data['img_metas'][0]
            if [int(meta['sequence_idx']) for meta in metas] != indices:
                raise RuntimeError('Validation data order differs from sampler order')
            streams = [sampler.stream_for_index[index] for index in indices]
            output = cache.predict(module, data, streams)
            results.extend(zip(indices, output))
            for index, stream in zip(indices, streams):
                if index in sampler.final_indices:
                    cache.states.pop(stream, None)
        torch.cuda.synchronize()
        inference_seconds = time.perf_counter() - start
        start_collect = time.perf_counter()
        if world_size > 1:
            if gpu_collect:
                results = collect_results_concat_in_memory(
                    results, size=len(data_loader.dataset), use_zip_sort=False)
            else:
                results = collect_results_cpu_concat_disk(
                    results, size=len(data_loader.dataset), tmpdir=None, use_zip_sort=False)
        if rank == 0:
            results = restore_result_order(results, len(data_loader.dataset))
        torch.cuda.synchronize()
        timing = dict(rank=rank, inference_seconds=inference_seconds,
                      collect_seconds=time.perf_counter() - start_collect,
                      batches=len(sampler), samples=len(sampler.indices),
                      requested_batch_per_gpu=sampler.batch_size)
        timings = [None] * world_size
        if world_size > 1:
            dist.all_gather_object(timings, timing)
        else:
            timings = [timing]
        model._streaming_val_timings = timings
        return results
    finally:
        cache.close()
