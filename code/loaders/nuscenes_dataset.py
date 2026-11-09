import os
import math

import numpy as np
from mmdet.datasets import DATASETS
from mmdet3d.datasets import NuScenesDataset
from pyquaternion import Quaternion

# The installed formatter explicitly reads NuScenesDataset.DefaultAttribute.
# Extend its runtime map as the original project fork did; no package file is changed.
NuScenesDataset.DefaultAttribute.setdefault('uav.drone', '')

import math
import numpy as np


@DATASETS.register_module()
class CustomNuScenesDataset(NuScenesDataset):
    DefaultAttribute = dict(NuScenesDataset.DefaultAttribute, **{'uav.drone': ''})
    
    def __init__(
        self,
        ann_file,
        pipeline=None,
        data_root=None,
        classes=None,
        load_interval=1,
        with_velocity=True,
        modality=None,
        test_mode=False,
        use_valid_flag=False,
        sequences_split_num=1,
        with_seq_flag=False,
        keep_consistent_seq_aug=True,
        **kwargs
    ):
        # 保存sequence相关参数
        self.sequences_split_num = sequences_split_num
        self.keep_consistent_seq_aug = keep_consistent_seq_aug
        self.current_aug = None
        self.last_id = None
        
        # Use the project's collected-UAV evaluation config and devkit.
        if classes is not None and list(classes) == ['uav.drone']:
            kwargs.setdefault('eval_version', 'detection_cvpr_uav')
        # 调用父类初始化
        super().__init__(
            ann_file=ann_file,
            pipeline=pipeline,
            data_root=data_root,
            classes=classes,
            load_interval=load_interval,
            with_velocity=with_velocity,
            modality=modality,
            test_mode=test_mode,
            use_valid_flag=use_valid_flag,
            **kwargs
        )
        
        # 如果启用sequence flag，设置sequence group flag
        if with_seq_flag:
            self._set_sequence_group_flag()

    def _evaluate_single(self, result_path, logger=None, metric='bbox', result_name='pts_bbox'):
        import mmcv
        from nuscenes import NuScenes
        from nuscenes.eval.detection.evaluate import NuScenesEval

        output_dir = os.path.dirname(result_path)
        eval_set_map = {'v1.0-mini': 'mini_val', 'v1.0-trainval': 'val', 'mid': 'val', 'full': 'val', '': 'val'}
        nusc = NuScenes(version=self.version, dataroot=self.data_root, verbose=False)
        evaluator = NuScenesEval(
            nusc, config=self.eval_detection_configs, result_path=result_path,
            eval_set=eval_set_map[self.version], output_dir=output_dir, verbose=False)
        evaluator.main(render_curves=False)
        metrics = mmcv.load(os.path.join(output_dir, 'metrics_summary.json'))
        prefix = result_name + '_NuScenes'
        detail = {}
        for name in self.CLASSES:
            for key, value in metrics['label_aps'][name].items():
                detail['%s/%s_AP_dist_%s' % (prefix, name, key)] = float('%.4f' % value)
            for key, value in metrics['label_tp_errors'][name].items():
                detail['%s/%s_%s' % (prefix, name, key)] = float('%.4f' % value)
        for key, value in metrics['tp_errors'].items():
            detail['%s/%s' % (prefix, self.ErrNameMapping[key])] = float('%.4f' % value)
        detail[prefix + '/NDS'] = metrics['nd_score']
        detail[prefix + '/mAP'] = metrics['mean_ap']
        return detail

    def _set_sequence_group_flag(self):
        """
        Set each sequence to be a different group
        为流式训练设置sequence group flag，确保同一batch中的样本来自同一sequence
        使用prev/next对应关系判断scene边界（与check_scene.py逻辑保持一致）
        """


        res = []

        # 加载NuScenes API
        from nuscenes.nuscenes import NuScenes
        nusc = NuScenes(version=self.version, dataroot=self.data_root, verbose=False)

        curr_sequence = 0

        for idx in range(len(self.data_infos)):
            info = self.data_infos[idx]
            curr_token = info.get("token", "")

            # 第一个样本总是sequence 0的开始
            if idx == 0:
                res.append(curr_sequence)
            else:
                # 检查prev/next对应关系
                is_new_scene = False

                try:
                    # 获取当前样本的prev字段
                    curr_sample = nusc.get("sample", curr_token)
                    curr_prev = curr_sample.get("prev", "")

                    # 获取前一个样本的token和next字段
                    prev_token = self.data_infos[idx - 1].get("token", "")
                    prev_sample = nusc.get("sample", prev_token)
                    prev_next = prev_sample.get("next", "")

                    # 判断条件：
                    # 1. 当前样本的prev应该等于前一个样本的token
                    # 2. 前一个样本的next应该等于当前样本的token
                    # 如果任何一个条件不满足，说明不连续，是新scene
                    if curr_prev != prev_token or prev_next != curr_token:
                        is_new_scene = True

                except Exception as e:
                    print(
                        f"警告: 无法验证样本 {idx} (token: {curr_token[:8] if curr_token else 'N/A'}) "
                        f"的prev/next关系: {e}"
                    )
                    # 如果无法验证，保守判断为新scene
                    is_new_scene = True

                if is_new_scene:
                    curr_sequence += 1

                res.append(curr_sequence)

        self.flag = np.array(res, dtype=np.int64)

        # -----------------------------
        # 下面是“切碎”逻辑：保留原模式 + 新增 target_len 模式
        # -----------------------------
        if self.sequences_split_num != 1:
            if self.sequences_split_num == "all":
                # 每个样本独立一个group
                self.flag = np.array(range(len(self.data_infos)), dtype=np.int64)

            elif self.sequences_split_num == "target_len":
                target_len = 10
                if target_len <= 0:
                    raise ValueError("self.sequences_split_target_len 必须为正整数")

                old_flag = self.flag
                max_old = int(old_flag.max()) if len(old_flag) > 0 else -1
                new_flag = np.empty_like(old_flag, dtype=np.int64)

                curr_new_flag = 0
                for curr_flag in range(max_old + 1):
                    idxs = np.where(old_flag == curr_flag)[0]
                    L = len(idxs)
                    if L == 0:
                        continue

                    # 自适应段数：Ki = ceil(L / target_len)
                    Ki = int(math.ceil(L / target_len))
                    if Ki <= 1:
                        new_flag[idxs] = curr_new_flag
                        curr_new_flag += 1
                        continue

                    # 让每段尽量等长，差不超过 1
                    base = L // Ki
                    rem = L % Ki  # 前 rem 段为 base+1

                    start = 0
                    for s in range(Ki):
                        seg_len = base + (1 if s < rem else 0)
                        seg_idxs = idxs[start:start + seg_len]
                        new_flag[seg_idxs] = curr_new_flag
                        curr_new_flag += 1
                        start += seg_len

                assert len(new_flag) == len(old_flag)
                self.flag = new_flag.astype(np.int64)

            else:
                # 原模式：固定切成 sequences_split_num 段
                # 注意：你原实现依赖 flag 在 data_infos 中连续；这里保持原样不改动
                bin_counts = np.bincount(self.flag)
                new_flags = []
                curr_new_flag = 0
                for curr_flag in range(len(bin_counts)):
                    curr_sequence_length = np.array(
                        list(
                            range(
                                0,
                                bin_counts[curr_flag],
                                math.ceil(bin_counts[curr_flag] / self.sequences_split_num),
                            )
                        )
                        + [bin_counts[curr_flag]]
                    )

                    for sub_seq_idx in (curr_sequence_length[1:] - curr_sequence_length[:-1]):
                        for _ in range(sub_seq_idx):
                            new_flags.append(curr_new_flag)
                        curr_new_flag += 1

                assert len(new_flags) == len(self.flag)
                assert (
                    len(np.bincount(new_flags))
                    == len(np.bincount(self.flag)) * self.sequences_split_num
                )
                self.flag = np.array(new_flags, dtype=np.int64)

    # debug
        # for i in range(len(self.flag)):
        #     print("flag", i, self.flag[i], self.data_infos[i].get('timestamp'))
        # print("flag: ", self.flag)
        # exit()

  
   
    def get_augmentation(self):
        """
        获取数据增强配置（用于keep_consistent_seq_aug）
        """
        if self.data_aug_conf is None:
            return None
        # 这里可以根据需要实现数据增强逻辑
        # 目前返回None，表示不使用数据增强
        return None

    def collect_sweeps(self, index, into_past=60, into_future=60):
        all_sweeps_prev = []
        curr_index = index
        while len(all_sweeps_prev) < into_past:
            curr_sweeps = self.data_infos[curr_index]['sweeps']
            if len(curr_sweeps) == 0:
                break
            all_sweeps_prev.extend(curr_sweeps)
            all_sweeps_prev.append(self.data_infos[curr_index - 1]['cams'])
            # all_sweeps_prev.append(self.data_infos[curr_index - 1]['sweeps'])
            curr_index = curr_index - 1
        
        all_sweeps_next = []
        curr_index = index + 1
        while len(all_sweeps_next) < into_future:
            if curr_index >= len(self.data_infos):
                break
            curr_sweeps = self.data_infos[curr_index]['sweeps']
            all_sweeps_next.extend(curr_sweeps[::-1])
            all_sweeps_next.append(self.data_infos[curr_index]['cams'])
            curr_index = curr_index + 1
       
        return all_sweeps_prev, all_sweeps_next

    def get_data_info(self, index):
        
        info = self.data_infos[index]
        sweeps_prev, sweeps_next = self.collect_sweeps(index)

        ego2global_translation = info['ego2global_translation']
        ego2global_rotation = info['ego2global_rotation']
        lidar2ego_translation = info['lidar2ego_translation']
        lidar2ego_rotation = info['lidar2ego_rotation']
        ego2global_rotation = Quaternion(ego2global_rotation).rotation_matrix
        lidar2ego_rotation = Quaternion(lidar2ego_rotation).rotation_matrix

        input_dict = dict(
            sample_idx=info['token'],
            sweeps={'prev': sweeps_prev, 'next': sweeps_next},
            timestamp=info['timestamp'] / 1e6,
            ego2global_translation=ego2global_translation,
            ego2global_rotation=ego2global_rotation,
            lidar2ego_translation=lidar2ego_translation,
            lidar2ego_rotation=lidar2ego_rotation,
        )
        
        # ========== 添加sequence相关信息，用于流式训练 ==========
        # 添加sequence_token，用于检测sequence边界
        if hasattr(self, 'flag'):
            input_dict['sequence_token'] = self.flag[index]
            input_dict['sequence_idx'] = index
        else:
            # 如果没有设置flag，使用scene_token或token作为sequence_token
            if 'scene_token' in info:
                input_dict['sequence_token'] = info['scene_token']
            else:
                input_dict['sequence_token'] = info['token']

        if self.modality['use_camera']:
            img_paths = []
            img_timestamps = []
            lidar2img_rts = []

            for _, cam_info in info['cams'].items():
                img_paths.append(os.path.relpath(cam_info['data_path']))
                img_timestamps.append(cam_info['timestamp'] / 1e6)

                # obtain lidar to image transformation matrix
                lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
                lidar2cam_t = cam_info['sensor2lidar_translation'] @ lidar2cam_r.T

                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t
                
                intrinsic = cam_info['cam_intrinsic']
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = (viewpad @ lidar2cam_rt.T)
                lidar2img_rts.append(lidar2img_rt)

            input_dict.update(dict(
                img_filename=img_paths,
                img_timestamp=img_timestamps,
                lidar2img=lidar2img_rts,
            ))
        if self.modality['use_radar']:
            pts = info['radar_sweeps']['radar_ms_pts']
            # pts =   BasePoints(pts, points_dim=6)    


            # pts = np.array(pts)
            # pts = sample_points(pts, 64)
            input_dict.update(dict(
                points=pts,
            ))
            input_dict.update(dict(
                valid_height=info['radar_sweeps']['valid_height'],
            ))
            # print("valid_height",info['radar_sweeps']['valid_height'])
            


        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos

        return input_dict
