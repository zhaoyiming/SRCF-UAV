from functools import partial
from mmcv.parallel import collate
from mmcv.runner import get_dist_info
from torch.utils.data import DataLoader
from mmdet.datasets.builder import worker_init_fn
from mmdet.datasets.samplers import DistributedGroupSampler, DistributedSampler, GroupSampler
from loaders.samplers.group_sampler import GroupInBatchBatchSampler, GroupInBatchValBatchSampler

def build_dataloader(dataset,
                     samples_per_gpu,
                     workers_per_gpu,
                     num_gpus=1,
                     dist=True,
                     shuffle=True,
                     seed=None,
                     use_streaming=False,
                     train=True,
                     **kwargs):

    rank, world_size = get_dist_info()
    batch_size = samples_per_gpu
    num_workers = workers_per_gpu

    if use_streaming:
        if train:
            batch_sampler = GroupInBatchBatchSampler(
                dataset, samples_per_gpu, world_size, rank, seed=seed)
            print("use streaming train batch sampler")
        else:
            if samples_per_gpu > 1 and world_size > 1:
                from streaming_val import ParallelStreamingValSampler
                batch_sampler = ParallelStreamingValSampler(
                    dataset, samples_per_gpu, world_size, rank, seed=seed)
            else:
                batch_sampler = GroupInBatchValBatchSampler(
                    dataset, 1, world_size, rank, seed=seed)
            print("use streaming val batch sampler")
        init_fn = partial(
                worker_init_fn, num_workers=num_workers, rank=rank,
                seed=seed) if seed is not None else None
        data_loader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,  # batch_sampler已经决定了batch组成
            num_workers=num_workers,
            collate_fn=partial(collate, samples_per_gpu=samples_per_gpu),
            pin_memory=False,
            worker_init_fn=init_fn,
            **kwargs
        )

    else:
        if dist:
            if shuffle:
                sampler = DistributedGroupSampler(
                    dataset, samples_per_gpu, world_size, rank, seed=seed)
            else:
                sampler = DistributedSampler(
                    dataset, world_size, rank, shuffle=False, seed=seed)
        else:
            sampler = GroupSampler(dataset, samples_per_gpu) if shuffle else None
            batch_size = num_gpus * samples_per_gpu
            num_workers = num_gpus * workers_per_gpu

        init_fn = partial(
            worker_init_fn, num_workers=num_workers, rank=rank,
            seed=seed) if seed is not None else None
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=partial(collate, samples_per_gpu=samples_per_gpu),
            pin_memory=False,
            worker_init_fn=init_fn,
            **kwargs)


    

    return data_loader
