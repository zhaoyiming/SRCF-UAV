"""Choose a full-validation batch on the first scheduled evaluation."""
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from loaders.builder import build_dataloader
from loaders.apis.test import multi_gpu_test_concat_fast


def choose_trial(trials):
    reference = trials[0]
    eligible = [trial for trial in trials if trial['equivalent_metrics']]
    return min(eligible or [reference], key=lambda trial: trial['total_validation_seconds'])


def select_validation_loader(runner, dataset, settings):
    rank, world = dist.get_rank(), dist.get_world_size()
    batches = list(settings['batches'])
    if not batches or batches[0] != 1 or any(batch < 1 for batch in batches):
        raise ValueError('Validation search must start with batch 1')
    output = Path(runner.work_dir) / 'val_tuning'
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
    random_state = random.getstate()
    numpy_state = np.random.get_state()
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state()
    stack = runner.model.module.pts_bbox_head.query_stack
    report = dict(epoch=runner.epoch + 1, samples=len(dataset), world_size=world,
                  backbone_depth=runner.model.module.img_backbone.depth, trials=[])

    def make_loader(batch):
        return build_dataloader(
            dataset, samples_per_gpu=batch,
            workers_per_gpu=settings.get('workers_per_gpu', 4), num_gpus=world,
            dist=True, shuffle=False, train=False,
            seed=settings.get('seed', 42), use_streaming=True,
            **settings.get('loader_options', {}))

    try:
        for batch in batches:
            stack.reset()
            loader = make_loader(batch)
            runner.logger.info('VAL_BATCH_SEARCH_START batch_per_gpu=%d', batch)
            dist.barrier()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            results = multi_gpu_test_concat_fast(runner.model, loader, gpu_collect=True)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - started
            local = dict(rank=rank, inference_and_collect_seconds=seconds,
                         peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30)
            ranks = [None] * world
            dist.all_gather_object(ranks, local)
            trial = None
            if rank == 0:
                try:
                    started = time.perf_counter()
                    metrics = dataset.evaluate(
                        results, jsonfile_prefix=str(output / ('bs%d' % batch)))
                    metric_seconds = time.perf_counter() - started
                    map_score = float(metrics['pts_bbox_NuScenes/mAP'])
                    nds = float(metrics['pts_bbox_NuScenes/NDS'])
                    reference = report['trials'][0] if report['trials'] else None
                    map_difference = abs(map_score - reference['mAP']) if reference else 0.
                    nds_difference = abs(nds - reference['NDS']) if reference else 0.
                    tolerance = settings.get('tolerance', .003)
                    trial = dict(
                        batch_per_gpu=batch, ranks=ranks, metrics_seconds=metric_seconds,
                        total_validation_seconds=max(x['inference_and_collect_seconds'] for x in ranks) + metric_seconds,
                        mAP=map_score, NDS=nds, mAP_difference=map_difference,
                        NDS_difference=nds_difference,
                        equivalent_metrics=map_difference <= tolerance and nds_difference <= tolerance)
                except Exception as error:
                    trial = dict(error=repr(error))
            shared = [trial]
            dist.broadcast_object_list(shared, src=0)
            trial = shared[0]
            iterator = getattr(loader, '_iterator', None)
            if iterator is not None:
                iterator._shutdown_workers()
            del loader, results
            stack.reset()
            if 'error' in trial:
                raise RuntimeError('Validation batch search failed: ' + trial['error'])
            report['trials'].append(trial)
            if rank == 0:
                (output / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
                runner.logger.info('VAL_BATCH_SEARCH_RESULT %s', json.dumps(trial))
        best = choose_trial(report['trials'])
        report.update(selected_batch_per_gpu=best['batch_per_gpu'],
                      speedup=report['trials'][0]['total_validation_seconds'] / best['total_validation_seconds'])
        if rank == 0:
            (output / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
            runner.logger.info('VAL_BATCH_SELECTED batch_per_gpu=%d speedup=%.3f',
                               best['batch_per_gpu'], report['speedup'])
        dist.barrier()
        return make_loader(best['batch_per_gpu'])
    finally:
        stack.reset()
        random.setstate(random_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(cpu_state)
        torch.cuda.set_rng_state(cuda_state)
