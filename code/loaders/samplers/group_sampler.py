import math
import itertools
from re import T
import numpy as np
import torch
from torch.utils.data import Sampler
from mmcv.runner import get_dist_info
import torch.distributed as dist
import logging
from collections import deque
import sys


def sync_random_seed(seed=None, device=None):
    rank, world_size = get_dist_info()
    if world_size == 1:
        return int(seed)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # broadcast seed from rank0
    t = torch.tensor(int(seed), dtype=torch.int64, device=device) if rank == 0 \
        else torch.tensor(0, dtype=torch.int64, device=device)

    if dist.is_available() and dist.is_initialized():
        dist.broadcast(t, src=0)

    return int(t.item())


class GroupInBatchBatchSampler(Sampler):
    """
    Finite-per-epoch batch sampler (fix: stop each epoch under EpochBasedRunner).
    Each iteration yields a LIST of indices (len == batch_size).
    Slot i always draws from its own group buffer (strict slot alignment).
    """

    def __init__(self,
                 dataset,
                 samples_per_gpu=1,
                 num_replicas=None,
                 rank=None,
                 seed=0,
                 skip_prob=0.0,
                 sequence_flip_prob=0.0,
                 keep_consistent_seq_aug=False,
                 ):
        _rank, _num_replicas = get_dist_info()
        if num_replicas is None:
            num_replicas = _num_replicas
        if rank is None:
            rank = _rank

        self.dataset = dataset
        self.batch_size = samples_per_gpu
        self.num_replicas = num_replicas
        self.rank = rank
        self.base_seed = sync_random_seed(seed)
        self.seed = self.base_seed
        self.size = len(dataset)

        self.skip_prob = skip_prob
        self.sequence_flip_prob = sequence_flip_prob
        self.keep_consistent_seq_aug = keep_consistent_seq_aug

        assert hasattr(dataset, 'flag')
        self.flag = dataset.flag
        self.group_sizes = np.bincount(self.flag)
        self.groups_num = len(self.group_sizes)
        
        # 初始化 epoch（需要在调试打印之前）
        self.epoch = 0
        self.global_batch_size = self.batch_size * self.num_replicas
        assert self.groups_num >= self.global_batch_size

        self.group_idx_to_sample_idxs = {
            g: np.where(self.flag == g)[0].tolist()
            for g in range(self.groups_num)
        }

        # 使用 stride 分配模式
        self.group_indices_per_local_slot = []
        self.buffer_per_local_slot = []
        self.np_rng = None  # 在set_epoch中初始化
        self.num_steps = math.ceil(self.size / self.global_batch_size)
        self.sampler = self          # 关键：兼容 mmcv SamplerSeedHook
        self.set_epoch(0)   


    def set_epoch(self, epoch):
        self.epoch = int(epoch)
        self.seed = self.base_seed + self.epoch  # large prime stride
        self.np_rng = np.random.RandomState(self.seed)
        self.buffer_per_local_slot = [deque() for _ in range(self.batch_size)]

        # 预先计算每个slot在本epoch的完整分配序列，确保显示和实际分配一致
        self.slot_allocation_sequences = []
        for local_slot in range(self.batch_size):
            global_slot_idx = self.rank * self.batch_size + local_slot
            temp_gen = self._group_indices_per_global_sample_idx(global_slot_idx)
            slot_groups = []
            for _ in range(self.num_steps):
                try:
                    group_id = next(temp_gen)
                    slot_groups.append(group_id)
                except StopIteration:
                    break
            self.slot_allocation_sequences.append(slot_groups)

        self.group_indices_per_local_slot = [
            iter(self.slot_allocation_sequences[local_slot])  # 使用预计算的序列
            for local_slot in range(self.batch_size)
        ]

        for i in range(len(self.slot_allocation_sequences)):
            print(f"slot_allocation_sequences, rank {self.rank}, slot{i}: {self.slot_allocation_sequences[i]}")
        

    def _infinite_group_indices(self):
        g = torch.Generator()
        g.manual_seed(self.seed)
        while True:
            perm = torch.randperm(self.groups_num, generator=g).tolist()
            yield from perm

    def _group_indices_per_global_sample_idx(self, global_sample_idx):
        yield from itertools.islice(self._infinite_group_indices(),
                                    global_sample_idx,
                                    None,
                                    self.global_batch_size)

    def __iter__(self):
        for _ in range(self.num_steps):
            curr_batch = []
            for local_slot in range(self.batch_size):
                if len(self.buffer_per_local_slot[local_slot]) == 0:
                    new_group = next(self.group_indices_per_local_slot[local_slot])
                    buf = self.group_idx_to_sample_idxs[new_group].copy()
                    if self.sequence_flip_prob > 0 and self.np_rng.uniform() < self.sequence_flip_prob:
                        buf = buf[::-1]
                    self.buffer_per_local_slot[local_slot] = deque(buf)


                if self.skip_prob > 0 and len(self.buffer_per_local_slot[local_slot]) > 1:
                    if self.np_rng.uniform() < self.skip_prob:
                        self.buffer_per_local_slot[local_slot].popleft()

                assert len(self.buffer_per_local_slot[local_slot]) > 0, f"Buffer for slot {local_slot} is empty unexpectedly. This should not happen. Check skip_prob and buffer filling logic."
                idx = self.buffer_per_local_slot[local_slot].popleft()
                curr_batch.append(idx)
            yield curr_batch
        

    def __len__(self):
        return self.num_steps

    def analyze_sample_coverage(self, samples_list, dataset, description=""):
        """分析样本覆盖情况的通用方法

        Args:
            samples_list: 样本列表的列表，例如 [samples_rank0, samples_rank1, ...]
            dataset: 数据集对象
            description: 描述信息

        Returns:
            dict: 包含各种统计信息的字典
        """
        # 合并所有rank的样本
        all_samples = []
        for samples in samples_list:
            for sample in samples:
                all_samples.extend(sample)

        unique_samples = set(all_samples)
        total_dataset_samples = len(dataset)

        # 计算覆盖率
        coverage_ratio = len(unique_samples) / total_dataset_samples * 100

        # 检查重复采样
        from collections import Counter
        sample_counts = Counter(all_samples)
        duplicates = {sample: count for sample, count in sample_counts.items() if count > 1}

        # 打印结果
        if description:
            print(f"   {description}")
        print(f"   合并后总样本数: {len(all_samples)}")
        print(f"   Unique样本数: {len(unique_samples)}")
        print(f"   数据集总样本数: {total_dataset_samples}")
        print(f"   覆盖率: {len(unique_samples)}/{total_dataset_samples} ({coverage_ratio:.1f}%)")

        if duplicates:
            print(f"   重复采样的样本: {len(duplicates)}个")
            print(f"   示例重复: {dict(list(duplicates.items())[:3])}")
        else:
            print(f"   无重复采样 ✓")

        return {
            'total_samples': len(all_samples),
            'unique_samples': len(unique_samples),
            'dataset_size': total_dataset_samples,
            'coverage_ratio': coverage_ratio,
            'duplicates': duplicates
        }



class GroupInBatchValBatchSampler(Sampler):
    """
    No-padding, deterministic, distributed-safe VAL sampler (batch_sampler style):
      - yields LIST of indices (len == batch_size)
      - batch_size must be 1
      - uses dataset.flag to group sequences (streaming correctness)
      - assigns sequences to ranks by CONTIGUOUS CHUNK on global sample order
        (Sparse4D-style contiguous split), not by round-robin.
      - sequences are NOT split across ranks (whole sequence assigned).
      - NOTE: per-rank sample counts may differ (allowed).
    """

    def __init__(
        self,
        dataset,
        samples_per_gpu=1,
        num_replicas=None,
        rank=None,
        seed=0,
        skip_prob=0.5,
        sequence_flip_prob=0.1,
        keep_consistent_seq_aug=False,
    ):
        _rank, _num_replicas = get_dist_info()
        if num_replicas is None:
            num_replicas = _num_replicas
        if rank is None:
            rank = _rank

        self.dataset = dataset
        self.batch_size = int(samples_per_gpu)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)

        assert self.batch_size == 1, "No-padding val/test requires samples_per_gpu=1."
        self.size = len(dataset)
        assert hasattr(dataset, "flag"), "dataset must have `flag` for sequence grouping"
        self.flag = dataset.flag

        sequence_groups = {}
        for idx in range(self.size):
            seq_id = int(self.flag[idx])
            sequence_groups.setdefault(seq_id, []).append(idx)
        sequence_splits = list(sequence_groups.values())
        
     
 
        base = self.size // self.num_replicas
        start = self.rank * base
        end = (self.rank + 1) * base if self.rank != self.num_replicas - 1 else self.size


        indices = []
        prefix = 0  # number of samples seen so far in global order

        for seq in sequence_splits:
            seq_len = len(seq)
            next_prefix = prefix + seq_len
            if prefix < end and next_prefix > start:
                # Sequence 与当前 rank 有重叠
                if prefix >= start:
                    # Sequence 在当前 rank 范围内，分配给当前 rank
                    indices.extend(seq)
                else:
                    # Sequence 跨越边界：prefix < start 但 next_prefix > start
                    # 计算 sequence 应该属于哪个 rank（第一个与其重叠的 rank）
                    # 如果 prefix < start，计算它应该属于哪个 rank：prefix // base
                    if base > 0:
                        assigned_rank = prefix // base
                    else:
                        # 边界情况：num_replicas > size，base=0
                        # 这种情况下，sequence应该分配给rank 0（第一个rank）
                        assigned_rank = 0
                    
                    if assigned_rank == self.rank:
                        # 当前 rank 是第一个应该处理它的 rank，分配
                        indices.extend(seq)

            prefix = next_prefix

        self.indices = indices
        self.num_steps = len(self.indices)

        self.sampler = self
        self.epoch = 0  # 初始化 epoch

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        for idx in self.indices:
            yield [idx]

    def __len__(self):
        return self.num_steps
