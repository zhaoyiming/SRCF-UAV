# Generate info files manually
import os
import mmcv
import tqdm
import pickle
import argparse
import numpy as np
from nuscenes import NuScenes
from pyquaternion import Quaternion
from readpcd import RadarPointCloud
import os.path as osp
from functools import reduce
from nuscenes.utils.geometry_utils import transform_matrix
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import torch
from typing import List
import matplotlib
matplotlib.use('Agg')
from readpcd import RadarPointCloud
from nuscenes.utils.geometry_utils import view_points, transform_matrix
np.random.seed(42)

import numpy as np


def sample_radar_points_old(radar_points, num_points, seed=None):
    """
    严格复刻旧逻辑：
      - N >= num_points: 直接打乱后取前 num_points
      - N <  num_points: 先拼接 [0..N-1]，再用 randint(0,N) 以“有放回”方式补齐到 num_points，
                         最后对索引整体打乱
    返回:
      sampled: (num_points, C)
      mask:    (num_points,)  全 1（均为真实点，可能有重复）
    """
    rng = np.random.RandomState(seed) if seed is not None else np.random
    pts = np.asarray(radar_points, dtype=np.float32)
    N = pts.shape[0]

    # 处理没有点的极端情况：维度未知时默认 C=0
    if N == 0:
        C = pts.shape[1] if pts.ndim == 2 else 0
        print("No radar points")
        return np.zeros((num_points, C), dtype=np.float32), np.zeros((num_points,), dtype=np.float32)

    if N >= num_points:
        idx = rng.permutation(N)[:num_points]
    else:
        base = np.arange(N)                                   # [0, 1, ..., N-1]
        extra = rng.randint(0, N, size=(num_points - N,))     # 有放回补样
        idx = np.concatenate([base, extra], axis=0)
        idx = rng.permutation(idx)                            # 整体再打乱

    sampled = pts[idx]
    mask = np.ones((num_points,), dtype=np.float32)           # 都是有效点（可能重复）
    return sampled, mask


def sample_radar_points(radar_points, num_points, seed=None):
    rng = np.random.RandomState(seed) if seed is not None else np.random
    pts = np.array(radar_points).astype(np.float32)  # (N, C)
    N = pts.shape[0]

    sampled = np.zeros((num_points, pts.shape[1]), dtype=np.float32)
    mask = np.zeros((num_points,), dtype=np.float32)

    if N == 0:
        print("No radar points")
        return sampled, mask

    # --- Step 1: 分类 ---
    # 假设速度在第 4,5 维 (vx, vy)，如果不是请告诉我改位置
    has_vel = np.linalg.norm(pts[:, 4:6], axis=1) > 1e-3
    valid_vel = pts[has_vel]
    zero_vel = pts[~has_vel]

    # --- Step 2: 采样 ---
    chosen = []
    if len(valid_vel) >= num_points:
        idx = rng.permutation(len(valid_vel))[:num_points]
        chosen = valid_vel[idx]
    else:
        chosen = valid_vel.tolist()
        rest_num = num_points - len(valid_vel)
        if len(zero_vel) > 0:
            idx2 = rng.permutation(len(zero_vel))[:min(rest_num, len(zero_vel))]
            chosen.extend(zero_vel[idx2])
        # 剩余的会由下面的补零逻辑处理

    chosen = np.array(chosen, dtype=np.float32)
    sampled[:len(chosen)] = chosen
    mask[:len(chosen)] = 1  # 前面真实点打 mask=1，其余 0

    return sampled, mask



def rotate_velocity(velocities, T_sensor_to_global):
    """
    Rotate velocities from sensor to global frame.

    Args:
        velocities: (N, 2) array, each row is (vx, vy) in sensor frame.
        T_sensor_to_global: (4, 4) homogeneous transform matrix.

    Returns:
        (N, 2) array of rotated velocities in global frame.
    """
    v3 = np.hstack([velocities, np.zeros((velocities.shape[0], 1))]).T  # (3, N)
    R = T_sensor_to_global[:3, :3]  # (3, 3)
    v_global = R @ v3  # (3, N)
    return v_global[:2, :].T


def proj2im(nusc, pc_cam, cam_token, min_z = 0.1):            
    cam_data = nusc.get('sample_data', cam_token) 
    cs_rec = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])         
    depth = pc_cam.points[2]    
    msk = pc_cam.points[2] >= min_z       
    points = view_points(pc_cam.points[:3, :], np.array(cs_rec['camera_intrinsic']).reshape(3,3), normalize=True)        
    x, y = points[0], points[1]
    msk =  reduce(np.logical_and, [x>0, x<1920, y>0, y<1080, msk])        
    return x, y, depth, msk    

def normalize_points(bboxes: np.ndarray, pc_range: List[float], filter_border=False):
    
        xyz = bboxes[..., 0:3].copy()
        if pc_range is not None:
            xyz[..., 0] = (xyz[..., 0] - pc_range[0]) / (pc_range[3] - pc_range[0])
            xyz[..., 1] = (xyz[..., 1] - pc_range[1]) / (pc_range[4] - pc_range[1])
            xyz[..., 2] = (xyz[..., 2] - pc_range[2]) / (pc_range[5] - pc_range[2])
        
        bboxes[..., :3]=xyz
        return bboxes


def cluster_radar_points_5d(points: np.ndarray, eps: float, min_samples: int) -> np.ndarray:


    if points.shape[1] < 6:
        raise ValueError("输入数组的列数必须至少为6 (x, y, z, rcs, vx, vy)。")

    # 步骤1: 提取x, y, z, vx, vy五个维度的特征
    # 使用np.c_来优雅地拼接列
    features_5d = np.c_[points[:, 0:3], points[:, 4:6]]

    # 步骤2: 特征缩放 (至关重要)
    # 创建一个StandardScaler对象
    scaler = StandardScaler()
    # 对特征进行拟合和转换，使其均值为0，方差为1
    features_5d_scaled = scaler.fit_transform(features_5d)

    # 步骤3: 在缩放后的数据上应用DBSCAN
    # eps的值需要针对标准化数据进行调整，通常在0.5-1.5之间
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = dbscan.fit_predict(features_5d_scaled)

    # 步骤4: 将聚类标签附加到原始数据
    points_with_clusters = np.hstack((points, cluster_labels[:, np.newaxis]))
    unique_labels_5d = np.unique(points_with_clusters[:, 6])

    cluster_center_info=[]
    for label in unique_labels_5d:
        # 如果标签不是噪声
        if label != -1:
            # 1. 筛选出所有属于当前标签的点
            cluster_mask = points_with_clusters[:, 6] == label
            points_in_cluster = points_with_clusters[cluster_mask]
            
            # 2. 计算这些点的平均位置和平均速度
            mean_pos = np.mean(points_in_cluster[:, 0:3], axis=0)
            mean_vel = np.mean(points_in_cluster[:, 4:6], axis=0)
    
            # # 3. 打印出这个计算出的“中心”信息
            # print(f"聚类 {int(label)}: ...")
            # print(f"  平均位置 (x,y,z) ≈ ({mean_pos[0]:.1f}, {mean_pos[1]:.1f}, {mean_pos[2]:.1f})")
            # print(f"  平均速度 (vx,vy) ≈ ({mean_vel[0]:.2f}, {mean_vel[1]:.2f}) m/s")
            cluster_center_info.append([mean_pos[0],mean_pos[1],mean_pos[2],mean_vel[0],mean_vel[1]])
    cluster_center_info=np.array(cluster_center_info)
    # print("cluster_center_info",cluster_center_info)
    # exit()

    return points_with_clusters,cluster_center_info




parser = argparse.ArgumentParser()
parser.add_argument('--data-root', default='data/nuscenes')
parser.add_argument('--version', default='')
args = parser.parse_args()


def get_cam_info(nusc, sample_data):
    pose_record = nusc.get('ego_pose', sample_data['ego_pose_token'])
    cs_record = nusc.get('calibrated_sensor', sample_data['calibrated_sensor_token'])
    
    sensor2ego_translation = cs_record['translation']
    ego2global_translation = pose_record['translation']
    sensor2ego_rotation = Quaternion(cs_record['rotation']).rotation_matrix
    ego2global_rotation = Quaternion(pose_record['rotation']).rotation_matrix
    cam_intrinsic = np.array(cs_record['camera_intrinsic'])

    sensor2global_rotation = sensor2ego_rotation.T @ ego2global_rotation.T
    sensor2global_translation = sensor2ego_translation @ ego2global_rotation.T + ego2global_translation

    return {
        'data_path': os.path.join(args.data_root, sample_data['filename']),
        'sensor2global_rotation': sensor2global_rotation,
        'sensor2global_translation': sensor2global_translation,
        'cam_intrinsic': cam_intrinsic,
        'timestamp': sample_data['timestamp'],
    }


def add_sweep_info(nusc, sample_infos, radar_full_vel):
    # print("sample_infos",sample_infos)
    # exit()
    # print(sample_infos)
    # print(sample_infos['metainfo'])
    # exit()
    # print(sample_infos['data_list'][0])
    del_ids = []
    all_candidates_path=[]
    for curr_id in tqdm.tqdm(range(len(sample_infos['infos']))):
        sample = nusc.get('sample', sample_infos['infos'][curr_id]['token'])

        cam_types = [
            'CAM_FRONT'
        ]

        curr_cams = dict()
        for cam in cam_types:
            curr_cams[cam] = nusc.get('sample_data', sample['data'][cam])

        for cam in cam_types:
            sample_data = nusc.get('sample_data', sample['data'][cam])
            sweep_cam = get_cam_info(nusc, sample_data)
            sample_infos['infos'][curr_id]['cams'][cam].update(sweep_cam)

        # remove unnecessary
        for cam in cam_types:
            del sample_infos['infos'][curr_id]['cams'][cam]['sample_data_token']
            del sample_infos['infos'][curr_id]['cams'][cam]['sensor2ego_translation']
            del sample_infos['infos'][curr_id]['cams'][cam]['sensor2ego_rotation']
            del sample_infos['infos'][curr_id]['cams'][cam]['ego2global_translation']
            del sample_infos['infos'][curr_id]['cams'][cam]['ego2global_rotation']

        sweep_infos = []
        if sample['prev'] != '':  # add sweep frame between two key frame
            for _ in range(5):
                sweep_info = dict()
                for cam in cam_types: 
                    if curr_cams[cam]['prev'] == '':    
                        sweep_info = sweep_infos[-1] 
                        break
                    sample_data = nusc.get('sample_data', curr_cams[cam]['prev'])
                    sweep_cam = get_cam_info(nusc, sample_data)
                    curr_cams[cam] = sample_data
                    sweep_info[cam] = sweep_cam
                sweep_infos.append(sweep_info)

        sample_infos['infos'][curr_id]['sweeps'] = sweep_infos

        radar_sweeps_path, r2l_rots, r2l_trans, time_gaps, all_pc, keyframe_mask = make_multisweep_radar_data_simple(nusc, sample, sup_vel=False, chan='RADAR_FRONT', ref_chan='RADAR_FRONT', nsweeps=5, min_distance=0.05)
        tranform_mat=transform_matrix(r2l_trans[0], Quaternion(r2l_rots[0]), inverse=False)
        all_pc.transform(tranform_mat)
        all_pc.rotate_velocity(all_pc.points, tranform_mat)
        final_pc=all_pc.points.T[:, [0, 1, 2, 5, 8, 9]]

        valid_pc = remove_static_points(final_pc)
        valid_height = valid_pc[:, 2]
        condition = (valid_height <= 10) & (valid_height >= -5)
        valid_indices = np.where(condition)[0]
        # final_filter_point = valid_pc[valid_indices][:, [0, 1, 2, 4, 5]]
        final_filter_point = valid_pc[valid_indices][:, [0, 1, 2, 4, 5]]

            
  
        # use full vel
        if radar_full_vel is not None:
            now_token=sample_infos['infos'][curr_id]['token']
            keyframe_mask=np.array(keyframe_mask)
            if now_token not in radar_full_vel:
                print("error token", curr_id)
                del_ids.append(curr_id)
                continue
            elif radar_full_vel[now_token]['preds'].shape[0] != keyframe_mask.sum():
                print("error num correct", curr_id, radar_full_vel[now_token]['preds'].shape[0], keyframe_mask.sum())
                del_ids.append(curr_id)
                continue
            else:
                now_data=radar_full_vel[now_token]
                mask=now_data['mask'].astype(bool)
                # print("mask",mask)
                expanded_mask = np.zeros(final_pc.shape[0], dtype=bool) 
                expanded_mask[:len(mask)] = mask  
                preds=now_data['preds']
                preds_global=rotate_velocity(preds, tranform_mat)
                extended_final_pc = np.hstack([final_pc, np.zeros((final_pc.shape[0], 2), dtype=final_pc.dtype)])
                extended_final_pc[expanded_mask, 6:8] = preds_global[mask]

                valid_pc = remove_static_points(extended_final_pc)
                valid_height = valid_pc[:, 2]
                condition = (valid_height <= 10) & (valid_height >= -5)
                valid_indices = np.where(condition)[0]
                final_filter_point = valid_pc[valid_indices][:, [0, 1, 2, 5, 6, 7]]
                    
           

        if final_pc.shape[0] == 0:
            print("remove no radar", curr_id)
            del_ids.append(curr_id)
        final_pc, mask = sample_radar_points_old(final_pc, 64)
      
        # label_file_path="/path/to/dataset/data_6843_2D/all_labels/labels/"
        # label_jpg_file=osp.join(label_file_path, sample_infos['infos'][curr_id]['cams']['CAM_FRONT']['data_path'].replace('jpg','txt').split('/')[-1])
        # label_file=open(label_jpg_file,'r')
        # label_lines=label_file.readlines()
        # if len(label_lines)==0:
        #     del_ids.append(curr_id)
        #     print("remove no label", curr_id)
        #     continue
        # print("final_pc",final_pc)
        # print("final_filter_point",final_filter_point)
        # exit()
  
        sample_infos['infos'][curr_id]['radar_sweeps'] = {
            "radar_sweeps_path": radar_sweeps_path,
            "radar_sweeps_r2l_rot": r2l_rots,
            "radar_sweeps_r2l_trans": r2l_trans,
            "radar_sweeps_time_gap": time_gaps,
            "radar_ms_pts": final_pc,
            "valid_height": final_filter_point
        }

    
    # sample_infos['infos']=sample_infos['infos'][:50]
    sample_infos['infos'] = [
        info for i, info in enumerate(sample_infos['infos'])
        if i not in del_ids 
    ]

    return sample_infos


    
def remove_static_points(all_pc):
    return all_pc[np.linalg.norm(all_pc[:, 4:6], axis=1) > 1e-6]

def remove_repeat_points(all_pc):
    order = np.argsort(all_pc[:, 3])[::-1]
    sorted_arr = all_pc[order]

    # 2. 对前三列去重，保留第一次出现（此时即为该组第四列最大的行）
    _, unique_idx = np.unique(sorted_arr[:, :3], axis=0, return_index=True)

    # 3. 取出结果
    result = sorted_arr[unique_idx]
    return result


def make_multisweep_radar_data_simple(nusc, sample_rec, sup_vel=False, chan='RADAR_FRONT', ref_chan='RADAR_FRONT', nsweeps=5, min_distance=0.05):
    radar_sweeps, r2l_rots, r2l_trans, time_gaps, all_pc = [], [], [], [], []
    points = np.zeros((18, 0), dtype=np.float32)
    all_pc = RadarPointCloud(points)
    all_times = np.zeros((1, 0))

    # Get reference pose and timestamp.
    ref_sd_token = sample_rec['data'][ref_chan]
    ref_sd_rec = nusc.get('sample_data', ref_sd_token)
    ref_pose_rec = nusc.get('ego_pose', ref_sd_rec['ego_pose_token'])
    ref_cs_rec = nusc.get('calibrated_sensor', ref_sd_rec['calibrated_sensor_token'])
    ref_time = 1e-6 * ref_sd_rec['timestamp']

    # Homogeneous transform from ego car frame to reference frame.
    ref_from_car = transform_matrix(ref_cs_rec['translation'], Quaternion(ref_cs_rec['rotation']), inverse=True)

    # Homogeneous transformation matrix from global to _current_ ego car frame.
    car_from_global = transform_matrix(ref_pose_rec['translation'], Quaternion(ref_pose_rec['rotation']),
                                        inverse=True)

    # Aggregate current and previous sweeps.
    sample_data_token = sample_rec['data'][chan]
    current_sd_rec = nusc.get('sample_data', sample_data_token)

    keyframe_mask=[]
    for i in range(nsweeps):
        # Load up the pointcloud and remove points close to the sensor.
        current_pc = RadarPointCloud.from_file(osp.join(nusc.dataroot, current_sd_rec['filename']))
        current_pc.remove_close(min_distance)

        # Get past pose.
        current_pose_rec = nusc.get('ego_pose', current_sd_rec['ego_pose_token'])
        global_from_car = transform_matrix(current_pose_rec['translation'],
                                            Quaternion(current_pose_rec['rotation']), inverse=False)

        # Homogeneous transformation matrix from sensor coordinate frame to ego car frame.
        current_cs_rec = nusc.get('calibrated_sensor', current_sd_rec['calibrated_sensor_token'])
        car_from_current = transform_matrix(current_cs_rec['translation'], Quaternion(current_cs_rec['rotation']),
                                            inverse=False)
        
        # Fuse four transformation matrices into one and perform transform.
        trans_matrix = reduce(np.dot, [ref_from_car, car_from_global, global_from_car, car_from_current])
        current_pc.transform(trans_matrix)

        # Add time vector which can be used as a temporal feature.
        time_lag = ref_time - 1e-6 * current_sd_rec['timestamp']  # Positive difference.
        if sup_vel:
            current_pc.points[:, :2] += current_pc.points[:, 4:6] * time_lag

        times = time_lag * np.ones((1, current_pc.nbr_points()))
        all_times = np.hstack((all_times, times))

        # Merge with key pc.
        all_pc.points = np.hstack((all_pc.points, current_pc.points))
        radar_sweeps.append(osp.join(nusc.dataroot, current_sd_rec['filename']))
        r2l_rots.append(current_cs_rec['rotation'])
        r2l_trans.append(current_cs_rec['translation'])
        time_gaps.append(time_lag)

        # Abort if there are no previous sweeps.
        if current_sd_rec['prev'] == '':
            break
        else:
            current_sd_rec = nusc.get('sample_data', current_sd_rec['prev'])

        if i==0:
            keyframe_mask=np.hstack((keyframe_mask,np.ones((current_pc.nbr_points())))) 
        else:
            keyframe_mask=np.hstack((keyframe_mask,np.zeros((current_pc.nbr_points()))))
        # keyframe_mask=np.hstack((keyframe_mask,np.ones((current_pc.nbr_points()))))
        

        
    return radar_sweeps, r2l_rots, r2l_trans, time_gaps, all_pc, keyframe_mask



if __name__ == '__main__':
    args.data_root = 'data/nuscenes'
    version = 'mid'
    version_dir = osp.join(args.data_root, version)
    nusc = NuScenes(version, args.data_root)
    new_eval_file={"train":[],"val":[]}

    # use_full_vel=False
    # if use_full_vel:
    #     radar_full_vel_path="/path/to/code/rad-vel/predictions/res_mid.pt"
    #     radar_full_vel=torch.load(radar_full_vel_path)
    # else:
    #     radar_full_vel=None

    # sample_infos = pickle.load(open(os.path.join(args.data_root, version, 'ming_streamrc_infos_train.pkl'), 'rb'))
    # sample_infos = add_sweep_info(nusc, sample_infos, radar_full_vel)
    # # sample_infos = add_sweep_info(nusc, sample_infos, None)
    # mmcv.dump(sample_infos, os.path.join(args.data_root, version, 'streamrc', 'nuscenes_infos_train_mingfusion_sweep.pkl'))
    # for sample_info in sample_infos['infos']:   
    #     new_eval_file['train'].append(sample_info['token'])

    # sample_infos = pickle.load(open(os.path.join(args.data_root, version, 'ming_streamrc_infos_val.pkl'), 'rb'))
    # sample_infos = add_sweep_info(nusc, sample_infos, radar_full_vel)
    # mmcv.dump(sample_infos, os.path.join(args.data_root, version, 'streamrc', 'nuscenes_infos_val_mingfusion_sweep.pkl'))
    # for sample_info in sample_infos['infos']:
    #     new_eval_file['val'].append(sample_info['token'])

    # print(len(new_eval_file['train']),len(new_eval_file['val']))
    # with open(os.path.join(args.data_root, version, 'splits_samples_filter_for_eval.json'), 'w') as f:
    #     json.dump(new_eval_file, f)

    # test_sample_infos = pickle.load(open(os.path.join(args.data_root, 'nuscenes_infos_test.pkl'), 'rb'))
    # test_sample_infos = add_sweep_info(nusc, test_sample_infos, None)
    # mmcv.dump(test_sample_infos, os.path.join(args.data_root, 'nuscenes_infos_test_mingfusion_sweep.pkl'))
    # # for sample_info in test_sample_infos['infos']:
    # #     new_eval_file['test'].append(sample_info['token'])
    
    check_one_test= mmcv.load(os.path.join(args.data_root, version, 'streamrc', 'nuscenes_infos_val_mingfusion_sweep.pkl'))
    print(check_one_test['infos'][0])
    exit()

    # pkl_list=os.listdir("/path/to/dataset/data_record_6843/nuscense-like/dataset_build/nuscense-like-scene/scenes_pkl/")
    # for pkl_file in pkl_list:
    #     pkl_path=os.path.join("/path/to/dataset/data_record_6843/nuscense-like/dataset_build/nuscense-like-scene/scenes_pkl/", pkl_file)
    #     pkl_sample_infos=pickle.load(open(pkl_path, 'rb'))
    #     pkl_sample_infos=add_sweep_info(nusc, pkl_sample_infos, None)
    #     mmcv.dump(pkl_sample_infos, os.path.join("/path/to/dataset/data_record_6843/nuscense-like/dataset_build/nuscense-like-scene/scenes_pkl", pkl_file.replace('.pkl', '_mingfusion_sweep.pkl')))


   

