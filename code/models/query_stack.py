import torch
import torch.nn as nn
from mmcv.cnn.bricks.registry import PLUGIN_LAYERS
from mmcv.runner import get_dist_info

__all__ = ["QueryStack"]


@PLUGIN_LAYERS.register_module()
class QueryStack(nn.Module):
    """
    Simplified temporal query management module.
    Manages query features across frames for streaming models.
    
    Note: Current frame queries are provided externally (not initialized here).
    This module only manages historical queries passed between frames.
    
    Args:
        num_query: total number of queries (default: 900)
        embed_dims: feature dimension of queries
        num_temp_queries: number of queries to pass across frames (default: 256)
        confidence_decay: decay factor for historical query confidence (default: 0.6)
        max_time_interval: maximum time interval (seconds) to use historical queries (default: 2.0)
        default_time_interval: default time interval when no history (default: 0.2)
        use_topk_in_update: if True, use topk to select current queries in update(); 
                           if False, use fixed positions (first N queries) (default: True)
        debug: enable debug prints for time_interval and sequence changes (default: False)
    """
    
    def __init__(
        self,
        num_query: int = 900,
        embed_dims: int = 256,
        num_temp_queries: int = 256,
        confidence_decay: float = 0.6,
        max_time_interval: float = 2.0,
        default_time_interval: float = 0.2,
        use_topk_in_update: bool = False,
        debug: bool = False,
        individual_init: bool = False,
        update_stage: list = [],
        fusion_stage: list = [],
    ):
        super(QueryStack, self).__init__()
        self.num_query = num_query
        self.embed_dims = embed_dims
        self.num_temp_queries = num_temp_queries
        self.confidence_decay = confidence_decay
        self.max_time_interval = max_time_interval
        self.default_time_interval = default_time_interval
        self.use_topk_in_update = use_topk_in_update
        self.debug = debug
        self.fused_feature = None
        self.fused_bbox = None
        self.time_interval = None

        self.existing_query_fea = None
        self.existing_query_bbox = None
        self.historical_feature = None
        self.historical_bbox = None
        self.historical_metas = None
        self.mask = None



        self.update_stage = update_stage
        self.fusion_stage = fusion_stage
        self.individual_init = individual_init
        self.prev_flag = False
        # Statistics for tracking training progress
        self.stats = {
            'total_calls': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'mask_valid_rate': 0.0,
            'avg_time_interval': 0.0,
            'sequence_changes': 0,
            'last_sequence_token': None,
            'time_interval_distribution': {},
            'sequence_lengths': {},
            'successful_history_samples': 0,
            'total_samples_with_cache': 0,
            'streaming_time_intervals': [],
        }
        self.expected_interval = 0.2  # Can be adjusted based on actual data
        self.print_interval = 50  # Print stats every N calls (reduced frequency)

    def on_epoch_end(self):
        """
        Called at the end of each epoch to print streaming report and reset statistics.
        This method should be called by the training framework or user code.
        """
        # Print streaming report
        self.print_streaming_report()
        self.reset_stats()

    def reset(self):
        """Reset cached queries (e.g., at the start of a new sequence)."""
        self.historical_feature = None
        self.historical_bbox = None  # 新增：缓存bbox
        self.historical_metas = None
        self.mask = None
        self.fused_feature = None
        self.fused_bbox = None
        self.existing_query_fea = None
        self.existing_query_bbox = None
        self.prev_flag = False
    
    def print_streaming_report(self):
        """
        打印完整的streaming验证报告（在epoch结束时调用）
        """
        if not self.debug:
            return
        rank, _ = get_dist_info()
        if rank != 0:
            return

        stats = self.stats
        if stats['total_calls'] == 0:
            return

        print(f"\n=== Streaming Report ===")

        cache_hit_rate = stats['cache_hits'] / stats['total_calls'] * 100.0 if stats['total_calls'] > 0 else 0.0
        print(f"Cache Hit Rate: {cache_hit_rate:.1f}% ({stats['cache_hits']}/{stats['total_calls']})")

        if stats['total_samples_with_cache'] > 0:
            success_rate = stats['successful_history_samples'] / stats['total_samples_with_cache'] * 100.0
            status = "✅ Good" if success_rate > 90 else "⚠️ Partial" if success_rate > 70 else "❌ Poor"
            print(f"History Usage: {status} {success_rate:.1f}% ({stats['successful_history_samples']}/{stats['total_samples_with_cache']})")

        if len(stats['streaming_time_intervals']) > 0:
            avg_interval = sum(stats['streaming_time_intervals']) / len(stats['streaming_time_intervals'])
            print(f"Avg Time Interval: {avg_interval:.3f}s (expected: {self.expected_interval:.1f}s)")

        if stats['sequence_lengths']:
            num_seqs = len(stats['sequence_lengths'])
            total_frames = sum(stats['sequence_lengths'].values())
            avg_len = total_frames / num_seqs
            print(f"Sequences: {num_seqs}, Avg Length: {avg_len:.1f} frames")
    
    def reset_stats(self):
        """Reset statistics (call at the start of each epoch)."""
        self.stats = {
            'total_calls': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'mask_valid_rate': 0.0,
            'avg_time_interval': 0.0,
            'sequence_changes': 0,
            'last_sequence_token': None,
            'time_interval_distribution': {},
            'sequence_lengths': {},
            'successful_history_samples': 0,
            'total_samples_with_cache': 0,
            'streaming_time_intervals': [],
        }
    
    def print_stats(self, step=None):
        """Print statistics (only on rank 0 and when debug=True)."""
        if not self.debug:
            return
        rank, _ = get_dist_info()
        if rank != 0:
            return

        stats = self.stats
        if stats['total_calls'] == 0:
            return


        print(f"\n[QueryStack] Step: {step if step is not None else 'N/A'}")
        
        # Core streaming metrics
        cache_hit_rate = stats['cache_hits'] / stats['total_calls'] * 100.0 if stats['total_calls'] > 0 else 0.0
        print(f"Calls: {stats['total_calls']}, Cache Hit: {cache_hit_rate:.1f}%, Seq Changes: {stats['sequence_changes']}")

        if stats['total_samples_with_cache'] > 0:
            streaming_success_rate = stats['successful_history_samples'] / stats['total_samples_with_cache'] * 100.0
            print(f"History Usage: {streaming_success_rate:.1f}% ({stats['successful_history_samples']}/{stats['total_samples_with_cache']})")
        else:
            print("No valid cache yet")
        
        # Time interval analysis
        if len(stats['streaming_time_intervals']) > 0:
            intervals = torch.tensor(stats['streaming_time_intervals'])
            avg_interval = intervals.mean().item()
            std_interval = intervals.std().item()
            print(f"Time: avg={avg_interval:.3f}s (±{std_interval:.3f}s), expected={self.expected_interval:.1f}s")

            if stats['time_interval_distribution']:
                sorted_intervals = sorted(stats['time_interval_distribution'].items(), key=lambda x: x[1], reverse=True)[:1]
                if abs(sorted_intervals[0][0] - self.expected_interval) > 0.1:
                    print(f"Warn: Common interval {sorted_intervals[0][0]:.1f}s ≠ expected {self.expected_interval:.1f}s")
        
        # Sequence statistics
        if stats['sequence_lengths']:
            seq_lengths = list(stats['sequence_lengths'].values())
            num_sequences = len(seq_lengths)
            avg_length = sum(seq_lengths) / num_sequences if num_sequences > 0 else 0
            print(f"Sequences: {num_sequences}, Avg Length: {avg_length:.1f} frames")
    
        # =========================
    # Logging helpers (只做统计/打印，不做业务决策)
    # =========================
    def _log_before_get(self, metas):
        """统计：总调用次数、batch内sequence分布、sequence长度、sequence切换；以及周期性打印。"""
        stats = self.stats
        stats["total_calls"] += 1


        # Track sequence changes and batch sequence distribution (logging only)
        if metas is not None and "img_metas" in metas:
            current_sequence_tokens = [m.get("sequence_token", None) for m in metas["img_metas"]]
            if current_sequence_tokens and current_sequence_tokens[0] is not None:
                # sequence长度统计：记录每个sequence的总帧数
                for seq_token in current_sequence_tokens:
                    if seq_token is not None:
                        stats["sequence_lengths"].setdefault(seq_token, 0)
                        stats["sequence_lengths"][seq_token] += 1  # 累加该sequence的总帧数

                # sequence切换统计
                if stats["last_sequence_token"] is not None and \
                stats["last_sequence_token"] != current_sequence_tokens[0]:
                    stats["sequence_changes"] += 1
                    if self.debug:  # 只打印前3次变化
                        rank, _ = get_dist_info()
                        if rank == 0:
                            print(f"[QueryStack] Sequence change: {stats['last_sequence_token']} -> {current_sequence_tokens[0]}")

                stats["last_sequence_token"] = current_sequence_tokens[0]

        # 周期性打印
        if stats["total_calls"] % self.print_interval == 0:
            self.print_stats()


    def _log_cache_hit(self, time_interval):
        """统计：cache hit、mask成功率、连续/跳帧计数、time_interval分布。"""
        stats = self.stats
        stats["cache_hits"] += 1

        # 有缓存可用的样本数统计
        batch_size_actual = self.mask.shape[0]
        successful_samples = self.mask.sum().item()
        stats["total_samples_with_cache"] += batch_size_actual
        stats["successful_history_samples"] += successful_samples

        # 时间间隔统计（仅分析用）
        valid_mask = torch.logical_and(time_interval != 0, self.mask)
        stats["mask_valid_rate"] = valid_mask.float().mean().item() * 100.0
        valid_intervals = time_interval[valid_mask]
        if valid_intervals.numel() > 0:
            stats["avg_time_interval"] = valid_intervals.abs().mean().item()

            for interval in valid_intervals.detach().cpu().tolist():
                interval_abs = abs(interval)
                stats["streaming_time_intervals"].append(interval_abs)

                bin_key = round(interval_abs * 10) / 10  # 0.1s bin
                stats["time_interval_distribution"].setdefault(bin_key, 0)
                stats["time_interval_distribution"][bin_key] += 1


    # =========================
    # Data helpers (只做数据处理/返回值构造)
    # =========================
    def _has_valid_cache(self, batch_size, metas):
        return (
            self.historical_feature is not None
            and batch_size == self.historical_feature.shape[0]
            and metas is not None
        )


    def _compute_time_interval(self, metas):
        """计算逐样本 time_interval（秒），并对齐dtype。"""
        history_time = self.historical_metas["timestamp"].to(dtype=torch.float64)
        current_time = metas["timestamp"].to(dtype=torch.float64)
        time_interval = (current_time - history_time)  # 转换为秒

        return time_interval.to(dtype=self.historical_feature.dtype)

    def get(self, metas=None, device=None):
        """
        Get historical frame queries.
        Aligned with Sparse4D InstanceBank.get() implementation.

        Returns:
            temp_query_feature: [B, num_temp_queries, embed_dims] or None
            time_interval: [B]
        """

        current_timestamps = torch.tensor([m['img_timestamp'][0] for m in metas],
                                            dtype=torch.float64, device=device)
        batch_size = len(metas)

        metas_for_stack = {
            'timestamp': current_timestamps,
            'img_metas': metas,
        }
        # 1) logging: 仅统计与打印
        if self.debug or not getattr(self, 'skip_debug_statistics', False):
            self._log_before_get(metas_for_stack)
        if self._has_valid_cache(batch_size, metas_for_stack):
            self.time_interval = self._compute_time_interval(metas_for_stack)
            self.mask = torch.logical_and(self.time_interval >= 0, self.time_interval <= self.max_time_interval)
            if self.mask.all():
                self.existing_query_fea = self.historical_feature
                self.existing_query_bbox = self.historical_bbox
                self.fused_feature = self.historical_feature
                self.fused_bbox = self.historical_bbox
                if self.debug or not getattr(self, 'skip_debug_statistics', False):
                    self._log_cache_hit(self.time_interval)
                self.prev_flag = True
            else:
                self.stats["cache_misses"] += 1
                self.reset()
        else:
            self.stats["cache_misses"] += 1
            self.reset()


    def get_temp_query_feature(self):
        return self.fused_feature

    def get_temp_query_bbox(self):
        return self.fused_bbox

    def update(self, query_feature, query_bbox, confidence):
        if self.existing_query_fea is None:
            return query_feature, query_bbox
        
        # 处理denoise queries
        num_dn = 0
        if query_feature.shape[1] > self.num_query:
            num_dn = query_feature.shape[1] - self.num_query
            dn_query_feature = query_feature[:, :num_dn, :]
            dn_query_bbox = query_bbox[:, :num_dn, :]
            normal_query_feature = query_feature[:, num_dn:, :]
            normal_confidence = confidence[:, num_dn:, :]
            normal_query_bbox = query_bbox[:, num_dn:, :]
        else:
            dn_query_feature = None
            normal_query_feature = query_feature
            normal_confidence = confidence
            dn_query_bbox = None
            normal_query_bbox = query_bbox
        

        N = self.num_query - self.num_temp_queries
        
        if self.use_topk_in_update:
            assert normal_confidence.dim() == 3, "normal_confidence should be 3D"
            assert normal_confidence.min() < 0.0 or normal_confidence.max() > 1.0, "confidence should be raw logits (before sigmoid), not probabilities in [0,1]"
            conf_scores = normal_confidence.squeeze(-1)  # [B, num_query]
            conf_scores = conf_scores.sigmoid()  # [B, num_query]
            
            # 获取topk索引
            _, top_indices = torch.topk(conf_scores, N, dim=1)  # [B, N]
            
            # 扩展索引维度并选择feature和bbox
            feat_indices = top_indices.unsqueeze(-1).expand(-1, -1, normal_query_feature.shape[-1])  # [B, N, embed_dims]
            bbox_indices = top_indices.unsqueeze(-1).expand(-1, -1, normal_query_bbox.shape[-1])  # [B, N, 10]
            
            selected_feature = torch.gather(normal_query_feature, 1, feat_indices)  # [B, N, embed_dims]
            selected_bbox = torch.gather(normal_query_bbox, 1, bbox_indices)  # [B, N, 10]
        else:
            # 使用固定位置选择当前queries（前N个）
            selected_feature = normal_query_feature[:, :N, :]  # [B, N, embed_dims]
            selected_bbox = normal_query_bbox[:, :N, :]  # [B, N, 10]
        
        # Check for batch size mismatch and handle gracefully
        if self.existing_query_fea.shape[0] != selected_feature.shape[0]:
            print(f"Warning: existing_query_fea.shape: {self.existing_query_fea.shape}, selected_feature.shape: {selected_feature.shape}")
            print("Batch size mismatch detected. Resetting cache to avoid errors.")
            self.reset()
            return query_feature, query_bbox

        merged_feature = torch.cat([selected_feature, self.existing_query_fea], dim=1)
        merged_bbox = torch.cat([ selected_bbox, self.existing_query_bbox,], dim=1)
        if self.mask is not None:
            updated_feature = torch.where(
                self.mask[:, None, None], merged_feature, normal_query_feature
            )
            updated_bbox = torch.where(
                self.mask[:, None, None], merged_bbox, normal_query_bbox
            )
        else:
            # 如果mask未设置，使用当前queries（通常不会发生）
            updated_feature = normal_query_feature
            updated_bbox = normal_query_bbox

        self.fused_feature = updated_feature[:,-self.num_temp_queries:,:]
        if num_dn > 0:
            query_feature = torch.cat([dn_query_feature, updated_feature], dim=1)
            query_bbox = torch.cat([dn_query_bbox, updated_bbox], dim=1)
        else:
            query_feature = updated_feature
            query_bbox = updated_bbox

        return query_feature, query_bbox
    
    def cache(self, query_feature, query_bbox, confidence, metas):
        """
        Cache current frame queries for next frame.
        Selects top-k high-confidence queries to pass to next frame.
        
        Args:
            query_feature: [B, total_queries, embed_dims] - queries after all processing
                          total_queries可能包含denoise queries（训练时）
            confidence: [B, total_queries, num_cls] or [B, total_queries] - confidence scores
            metas: metadata dict containing 'timestamp' and 'img_metas'
            query_bbox: [B, total_queries, 10] - bbox to cache (需要与feature同步)
        """
        if self.num_temp_queries <= 0:
            return
        
        # Detach防止梯度传播
        query_feature = query_feature.detach()
        confidence = confidence.detach()
        query_bbox = query_bbox.detach()
        
        current_timestamps = torch.tensor([m['img_timestamp'][0] for m in metas],
                                            dtype=torch.float64, device=query_feature.device)
        metas_for_stack = {
            'timestamp': current_timestamps,
            'img_metas': metas,
        }

        total_queries = query_feature.shape[1]
        dn_pad_size = total_queries - self.num_query  # denoise queries的数量
        if dn_pad_size > 0:
            normal_query_feature = query_feature[:, dn_pad_size:, :]  # [B, num_query, embed_dims]
            normal_confidence = confidence[:, dn_pad_size:] 
            normal_query_bbox = query_bbox[:, dn_pad_size:, :]  # [B, num_query, 10]
        else:
            normal_query_feature = query_feature
            normal_confidence = confidence
            normal_query_bbox = query_bbox
        
        assert normal_confidence.dim() == 3, "normal_confidence should be 3D"
        # assert normal_confidence.min() < 0.0 or normal_confidence.max() > 1.0, "confidence should be raw logits (before sigmoid), not probabilities in [0,1]"
        conf_scores = normal_confidence.squeeze(-1)  # [B, num_query]
        conf_scores = conf_scores.sigmoid()  # [B, num_query]
        selected_confidence, top_indices = torch.topk(conf_scores, self.num_temp_queries, dim=1)  # [B, num_temp_queries]
        feat_indices = top_indices.unsqueeze(-1).expand(-1, -1, normal_query_feature.shape[-1])  # [B, num_temp_queries, embed_dims]
        bbox_indices = top_indices.unsqueeze(-1).expand(-1, -1, normal_query_bbox.shape[-1])  # [B, num_temp_queries, 10]
        self.historical_feature = torch.gather(normal_query_feature, 1, feat_indices)  # [B, num_temp_queries, embed_dims]
        self.historical_bbox = torch.gather(normal_query_bbox, 1, bbox_indices)  # [B, num_temp_queries, 10]
        self.historical_metas = metas_for_stack



