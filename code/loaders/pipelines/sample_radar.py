import torch
from mmdet.datasets.builder import PIPELINES
import numpy as np
from mmdet3d.core.points import BasePoints
# from mmdet3d.core.bbox import DistanceCompactness
from mmcv.parallel import DataContainer as DC     # <= MMCV‑1.x



@PIPELINES.register_module()
class PackRadarDC(object):
    def __call__(self, results):
        results['radar_pts'] = DC(torch.from_numpy(results['radar_pts']), stack=False)
        return results


@PIPELINES.register_module()
class SampleRadarPoints:
    def __init__(self, num_points=64, pts_dim=6):
        self.num = num_points
        self.pts_dim = pts_dim

    def __call__(self, results):
        # 1) 取原始雷达点
        if isinstance(results.get('points'), dict):
            pts = results['points']['points']
            # mask = results['points']['points_mask']
        else:
            pts = results['points']
            # mask = results['points_mask']
      
        pts=np.array(pts).astype(np.float32)
        assert pts.shape[0] == self.num, "pts.shape[0] != self.num"
        # N = pts.shape[0]
        # if N == 0:
        #     print("No radar points")
        #     return np.zeros((self.num, self.pts_dim))
        # if N >= self.num:
        #     idx = np.random.permutation(N)[:self.num]
        # else:
        #     extra = np.random.randint(0, N, size=(self.num - N,))
        #     idx = np.concatenate([np.arange(N), extra], axis=0)
        #     idx = np.random.permutation(idx)

        # sampled = pts[idx]  # shape (num, C)
        sampled_tensor = torch.tensor(pts, dtype=torch.float32)
        # sampled_mask = torch.tensor(mask, dtype=torch.float32)
        results['points'] = BasePoints(sampled_tensor, points_dim=self.pts_dim)
        # results['points_mask'] = BasePoints(sampled_mask, points_dim=1)

        return results

    def __repr__(self):
        return (f'{self.__class__.__name__}(num_points={self.num}, '
                f'pts_dim={self.pts_dim})')


@PIPELINES.register_module()
class SampleRadarPointsDeterministic:
    """确定式雷达点采样，仅按速度大小排序。"""
    def __init__(self, num_points=64, pts_dim=6):
        self.num = num_points
        self.pts_dim = pts_dim

    def __call__(self, results):
        # 取 points
        if isinstance(results['points'], dict):
            pts = results['points']['points']
        else:
            pts = results['points']
        pts = np.array(pts, dtype=np.float32)
        N = pts.shape[0]

        if N == 0:
            sampled = np.zeros((self.num, self.pts_dim), dtype=np.float32)
        else:
            # 按速度模长降序 + index 升序稳定排序
            vxvy = pts[:, [4, 5]]
            speed = np.linalg.norm(vxvy, axis=1)
            idx = np.arange(N, dtype=np.int64)
            order = np.lexsort((idx, -speed))

            if N >= self.num:
                sel = order[:self.num]
            else:
                reps = int(np.ceil(self.num / N))
                sel = np.tile(order, reps)[:self.num]

            sampled = pts[sel]

        results['points'] = BasePoints(torch.from_numpy(sampled), points_dim=self.pts_dim)
        return results
