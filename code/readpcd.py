# nuScenes dev-kit.
# Code written by Oscar Beijbom, 2018.

import copy
import os.path as osp
import struct
from abc import ABC, abstractmethod
from functools import reduce
from typing import Tuple, List, Dict

import cv2
import numpy as np
from matplotlib.axes import Axes
from pyquaternion import Quaternion

from nuscenes.lidarseg.lidarseg_utils import colormap_to_colors, create_lidarseg_legend
from nuscenes.utils.data_io import load_bin_file
from nuscenes.utils.geometry_utils import view_points, transform_matrix

def radial_velocity_projection(x, y, vr, eps=1e-6):
    """
    nuScenes-style radial velocity projection (fixed platform)
    Support batch processing with numpy arrays

    Args:
        x, y : np.ndarray or array-like
            Radar point coordinates in radar frame (meters)
        vr : np.ndarray or array-like
            Radial velocity (m/s), positive if moving away
        eps : float
            Small value to avoid division by zero

    Returns:
        vx, vy : np.ndarray
            Velocity components along x and y axes
    """
    
    x = np.asarray(x)
    y = np.asarray(y)
    vr = np.asarray(vr)

    r = np.sqrt(x*x + y*y) + eps
    ux = x / r
    uy = y / r

    vx = vr * ux
    vy = vr * uy

    # print("vx, vy", vr, x, y, vx, vy)
    return vx, vy

dtype = np.dtype({
    'names': ['x', 'y', 'z', 'intensity', 'velocity'],
    'formats': ['<f4', '<f4', '<f4', '<f4', '<f4'],
    'offsets': [0, 4, 8, 16, 20],
    'itemsize': 32
})


class PointCloud(ABC):
    """
    Abstract class for manipulating and viewing point clouds.
    Every point cloud (lidar and radar) consists of points where:
    - Dimensions 0, 1, 2 represent x, y, z coordinates.
        These are modified when the point cloud is rotated or translated.
    - All other dimensions are optional. Hence these have to be manually modified if the reference frame changes.
    """

    def __init__(self, points: np.ndarray):
        """
        Initialize a point cloud and check it has the correct dimensions.
        :param points: <np.float: d, n>. d-dimensional input point cloud matrix.
        """
        assert points.shape[0] == self.nbr_dims(), 'Error: Pointcloud points must have format: %d x n' % self.nbr_dims()
        self.points = points

    @staticmethod
    @abstractmethod
    def nbr_dims() -> int:
        """
        Returns the number of dimensions.
        :return: Number of dimensions.
        """
        pass
    
    @classmethod
    def rotate_velocity(cls, pointcloud, T_sensor_to_global):
        v = pointcloud[[8, 9], :]                  # (2, N)
        v3 = np.vstack((v, np.zeros_like(v[0, :])))  # (3, N)

        # 2. 只用旋转，不要平移
        R = T_sensor_to_global[:3, :3]             # (3, 3)
        v_global = R @ v3                          # (3, N)

        # 3. 回写（只保留全局 xy 分量）
        pointcloud[8, :] = v_global[0, :]
        pointcloud[9, :] = v_global[1, :]


    @classmethod
    def from_file(cls, file_path):
        """
        自定义从文件加载点云数据，返回MyRadarPointCloud实例
        假设你的文件是存储了N×4的float32二进制数据，
        顺序和RadarPointCloud一致：[x, y, z, velocity]
        """
        raw_flat_data = np.fromfile(file_path, dtype=dtype)
        temp_pts=np.zeros((raw_flat_data.shape[0], cls.nbr_dims()))
        # print(raw_flat_data['x'].shape)
        vx, vy = radial_velocity_projection(raw_flat_data['x'], raw_flat_data['y'], raw_flat_data['velocity'])

        temp_pts[:,0]=raw_flat_data['x']
        temp_pts[:,1]=raw_flat_data['y']
        temp_pts[:,2]=raw_flat_data['z']
        temp_pts[:,5]=raw_flat_data['intensity']
        temp_pts[:,6]=vx
        temp_pts[:,7]=vy
        temp_pts[:,8]=vx
        temp_pts[:,9]=vy
        temp_pts[:,-1]=raw_flat_data['velocity']
        # reshape成 (N, 18)
        points = temp_pts.reshape((-1, cls.nbr_dims()))

        # 创建实例，调用父类构造函数
        return cls(points.T)  # RadarPointCloud构造函数一般接收shape=(18, N)

    @classmethod
    def from_file_multisweep(cls,
                             nusc: 'NuScenes',
                             sample_rec: Dict,
                             chan: str,
                             ref_chan: str,
                             nsweeps: int = 5,
                             min_distance: float = 0.05) -> Tuple['PointCloud', np.ndarray]:
        """
        Return a point cloud that aggregates multiple sweeps.
        As every sweep is in a different coordinate frame, we need to map the coordinates to a single reference frame.
        As every sweep has a different timestamp, we need to account for that in the transformations and timestamps.
        :param nusc: A NuScenes instance.
        :param sample_rec: The current sample.
        :param chan: The lidar/radar channel from which we track back n sweeps to aggregate the point cloud.
        :param ref_chan: The reference channel of the current sample_rec that the point clouds are mapped to.
        :param nsweeps: Number of sweeps to aggregated.
        :param min_distance: Distance below which points are discarded.
        :return: (all_pc, all_times). The aggregated point cloud and timestamps.
        """
        # Init.
        points = np.zeros((cls.nbr_dims(), 0), dtype=np.float32 )
        all_pc = cls(points)
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
        for _ in range(nsweeps):
            # Load up the pointcloud and remove points close to the sensor.
            current_pc = cls.from_file(osp.join(nusc.dataroot, current_sd_rec['filename']))
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
            times = time_lag * np.ones((1, current_pc.nbr_points()))
            all_times = np.hstack((all_times, times))

            # Merge with key pc.
            all_pc.points = np.hstack((all_pc.points, current_pc.points))

            # Abort if there are no previous sweeps.
            if current_sd_rec['prev'] == '':
                break
            else:
                current_sd_rec = nusc.get('sample_data', current_sd_rec['prev'])

        return all_pc, all_times

    def nbr_points(self) -> int:
        """
        Returns the number of points.
        :return: Number of points.
        """
        return self.points.shape[1]

    def subsample(self, ratio: float) -> None:
        """
        Sub-samples the pointcloud.
        :param ratio: Fraction to keep.
        """
        selected_ind = np.random.choice(np.arange(0, self.nbr_points()), size=int(self.nbr_points() * ratio))
        self.points = self.points[:, selected_ind]

    def remove_close(self, radius: float) -> None:
        """
        Removes point too close within a certain radius from origin.
        :param radius: Radius below which points are removed.
        """

        x_filt = np.abs(self.points[0, :]) < radius
        y_filt = np.abs(self.points[1, :]) < radius
        not_close = np.logical_not(np.logical_and(x_filt, y_filt))
        self.points = self.points[:, not_close]

    def translate(self, x: np.ndarray) -> None:
        """
        Applies a translation to the point cloud.
        :param x: <np.float: 3, 1>. Translation in x, y, z.
        """
        for i in range(3):
            self.points[i, :] = self.points[i, :] + x[i]

    def rotate(self, rot_matrix: np.ndarray) -> None:
        """
        Applies a rotation.
        :param rot_matrix: <np.float: 3, 3>. Rotation matrix.
        """
        self.points[:3, :] = np.dot(rot_matrix, self.points[:3, :])

    def transform(self, transf_matrix: np.ndarray) -> None:
        """
        Applies a homogeneous transform.
        :param transf_matrix: <np.float: 4, 4>. Homogenous transformation matrix.
        """
        self.points[:3, :] = transf_matrix.dot(np.vstack((self.points[:3, :], np.ones(self.nbr_points()))))[:3, :]

    def render_height(self,
                      ax: Axes,
                      view: np.ndarray = np.eye(4),
                      x_lim: Tuple[float, float] = (-20, 20),
                      y_lim: Tuple[float, float] = (-20, 20),
                      marker_size: float = 1) -> None:
        """
        Very simple method that applies a transformation and then scatter plots the points colored by height (z-value).
        :param ax: Axes on which to render the points.
        :param view: <np.float: n, n>. Defines an arbitrary projection (n <= 4).
        :param x_lim: (min, max). x range for plotting.
        :param y_lim: (min, max). y range for plotting.
        :param marker_size: Marker size.
        """
        self._render_helper(2, ax, view, x_lim, y_lim, marker_size)

    def render_intensity(self,
                         ax: Axes,
                         view: np.ndarray = np.eye(4),
                         x_lim: Tuple[float, float] = (-20, 20),
                         y_lim: Tuple[float, float] = (-20, 20),
                         marker_size: float = 1) -> None:
        """
        Very simple method that applies a transformation and then scatter plots the points colored by intensity.
        :param ax: Axes on which to render the points.
        :param view: <np.float: n, n>. Defines an arbitrary projection (n <= 4).
        :param x_lim: (min, max).
        :param y_lim: (min, max).
        :param marker_size: Marker size.
        """
        self._render_helper(3, ax, view, x_lim, y_lim, marker_size)

    def _render_helper(self,
                       color_channel: int,
                       ax: Axes,
                       view: np.ndarray,
                       x_lim: Tuple[float, float],
                       y_lim: Tuple[float, float],
                       marker_size: float) -> None:
        """
        Helper function for rendering.
        :param color_channel: Point channel to use as color.
        :param ax: Axes on which to render the points.
        :param view: <np.float: n, n>. Defines an arbitrary projection (n <= 4).
        :param x_lim: (min, max).
        :param y_lim: (min, max).
        :param marker_size: Marker size.
        """
        points = view_points(self.points[:3, :], view, normalize=False)
        ax.scatter(points[0, :], points[1, :], c=self.points[color_channel, :], s=marker_size)
        ax.set_xlim(x_lim[0], x_lim[1])
        ax.set_ylim(y_lim[0], y_lim[1])




class RadarPointCloud(PointCloud):

    # Class-level settings for radar pointclouds, see from_file().
    invalid_states = [0]  # type: List[int]
    dynprop_states = range(7)  # type: List[int] # Use [0, 2, 6] for moving objects only.
    ambig_states = [3]  # type: List[int]

    @classmethod
    def disable_filters(cls) -> None:
        """
        Disable all radar filter settings.
        Use this method to plot all radar returns.
        Note that this method affects the global settings.
        """
        cls.invalid_states = list(range(18))
        cls.dynprop_states = list(range(8))
        cls.ambig_states = list(range(5))

    @classmethod
    def default_filters(cls) -> None:
        """
        Set the defaults for all radar filter settings.
        Note that this method affects the global settings.
        """
        cls.invalid_states = [0]
        cls.dynprop_states = range(7)
        cls.ambig_states = [3]

    @staticmethod
    def nbr_dims() -> int:
        """
        Returns the number of dimensions.
        :return: Number of dimensions.
        """
        return 18





class Box:
    """ Simple data class representing a 3d box including, label, score and velocity. """

    def __init__(self,
                 center: List[float],
                 size: List[float],
                 orientation: Quaternion,
                 label: int = np.nan,
                 score: float = np.nan,
                 velocity: Tuple = (np.nan, np.nan, np.nan),
                 name: str = None,
                 token: str = None):
        """
        :param center: Center of box given as x, y, z.
        :param size: Size of box in width, length, height.
        :param orientation: Box orientation.
        :param label: Integer label, optional.
        :param score: Classification score, optional.
        :param velocity: Box velocity in x, y, z direction.
        :param name: Box name, optional. Can be used e.g. for denote category name.
        :param token: Unique string identifier from DB.
        """
        assert not np.any(np.isnan(center))
        assert not np.any(np.isnan(size))
        assert len(center) == 3
        assert len(size) == 3
        assert type(orientation) == Quaternion

        self.center = np.array(center)
        self.wlh = np.array(size)
        self.orientation = orientation
        self.label = int(label) if not np.isnan(label) else label
        self.score = float(score) if not np.isnan(score) else score
        self.velocity = np.array(velocity)
        self.name = name
        self.token = token

    def __eq__(self, other):
        center = np.allclose(self.center, other.center)
        wlh = np.allclose(self.wlh, other.wlh)
        orientation = np.allclose(self.orientation.elements, other.orientation.elements)
        label = (self.label == other.label) or (np.isnan(self.label) and np.isnan(other.label))
        score = (self.score == other.score) or (np.isnan(self.score) and np.isnan(other.score))
        vel = (np.allclose(self.velocity, other.velocity) or
               (np.all(np.isnan(self.velocity)) and np.all(np.isnan(other.velocity))))

        return center and wlh and orientation and label and score and vel

    def __repr__(self):
        repr_str = 'label: {}, score: {:.2f}, xyz: [{:.2f}, {:.2f}, {:.2f}], wlh: [{:.2f}, {:.2f}, {:.2f}], ' \
                   'rot axis: [{:.2f}, {:.2f}, {:.2f}], ang(degrees): {:.2f}, ang(rad): {:.2f}, ' \
                   'vel: {:.2f}, {:.2f}, {:.2f}, name: {}, token: {}'

        return repr_str.format(self.label, self.score, self.center[0], self.center[1], self.center[2], self.wlh[0],
                               self.wlh[1], self.wlh[2], self.orientation.axis[0], self.orientation.axis[1],
                               self.orientation.axis[2], self.orientation.degrees, self.orientation.radians,
                               self.velocity[0], self.velocity[1], self.velocity[2], self.name, self.token)

    @property
    def rotation_matrix(self) -> np.ndarray:
        """
        Return a rotation matrix.
        :return: <np.float: 3, 3>. The box's rotation matrix.
        """
        return self.orientation.rotation_matrix

    def translate(self, x: np.ndarray) -> None:
        """
        Applies a translation.
        :param x: <np.float: 3, 1>. Translation in x, y, z direction.
        """
        self.center += x

    def rotate(self, quaternion: Quaternion) -> None:
        """
        Rotates box.
        :param quaternion: Rotation to apply.
        """
        self.center = np.dot(quaternion.rotation_matrix, self.center)
        self.orientation = quaternion * self.orientation
        self.velocity = np.dot(quaternion.rotation_matrix, self.velocity)

    def corners(self, wlh_factor: float = 1.0) -> np.ndarray:
        """
        Returns the bounding box corners.
        :param wlh_factor: Multiply w, l, h by a factor to scale the box.
        :return: <np.float: 3, 8>. First four corners are the ones facing forward.
            The last four are the ones facing backwards.
        """
        w, l, h = self.wlh * wlh_factor

        # 3D bounding box corners. (Convention: x points forward, y to the left, z up.)
        x_corners = l / 2 * np.array([1,  1,  1,  1, -1, -1, -1, -1])
        y_corners = w / 2 * np.array([1, -1, -1,  1,  1, -1, -1,  1])
        z_corners = h / 2 * np.array([1,  1, -1, -1,  1,  1, -1, -1])
        corners = np.vstack((x_corners, y_corners, z_corners))

        # Rotate
        corners = np.dot(self.orientation.rotation_matrix, corners)

        # Translate
        x, y, z = self.center
        corners[0, :] = corners[0, :] + x
        corners[1, :] = corners[1, :] + y
        corners[2, :] = corners[2, :] + z

        return corners

    def bottom_corners(self) -> np.ndarray:
        """
        Returns the four bottom corners.
        :return: <np.float: 3, 4>. Bottom corners. First two face forward, last two face backwards.
        """
        return self.corners()[:, [2, 3, 7, 6]]

    def render(self,
               axis: Axes,
               view: np.ndarray = np.eye(3),
               normalize: bool = False,
               colors: Tuple = ('b', 'r', 'k'),
               linewidth: float = 2) -> None:
        """
        Renders the box in the provided Matplotlib axis.
        :param axis: Axis onto which the box should be drawn.
        :param view: <np.array: 3, 3>. Define a projection in needed (e.g. for drawing projection in an image).
        :param normalize: Whether to normalize the remaining coordinate.
        :param colors: (<Matplotlib.colors>: 3). Valid Matplotlib colors (<str> or normalized RGB tuple) for front,
            back and sides.
        :param linewidth: Width in pixel of the box sides.
        """
        corners = view_points(self.corners(), view, normalize=normalize)[:2, :]

        def draw_rect(selected_corners, color):
            prev = selected_corners[-1]
            for corner in selected_corners:
                axis.plot([prev[0], corner[0]], [prev[1], corner[1]], color=color, linewidth=linewidth)
                prev = corner

        # Draw the sides
        for i in range(4):
            axis.plot([corners.T[i][0], corners.T[i + 4][0]],
                      [corners.T[i][1], corners.T[i + 4][1]],
                      color=colors[2], linewidth=linewidth)

        # Draw front (first 4 corners) and rear (last 4 corners) rectangles(3d)/lines(2d)
        draw_rect(corners.T[:4], colors[0])
        draw_rect(corners.T[4:], colors[1])

        # Draw line indicating the front
        center_bottom_forward = np.mean(corners.T[2:4], axis=0)
        center_bottom = np.mean(corners.T[[2, 3, 7, 6]], axis=0)
        axis.plot([center_bottom[0], center_bottom_forward[0]],
                  [center_bottom[1], center_bottom_forward[1]],
                  color=colors[0], linewidth=linewidth)

    def render_cv2(self,
                   im: np.ndarray,
                   view: np.ndarray = np.eye(3),
                   normalize: bool = False,
                   colors: Tuple = ((0, 0, 255), (255, 0, 0), (155, 155, 155)),
                   linewidth: int = 2) -> None:
        """
        Renders box using OpenCV2.
        :param im: <np.array: width, height, 3>. Image array. Channels are in BGR order.
        :param view: <np.array: 3, 3>. Define a projection if needed (e.g. for drawing projection in an image).
        :param normalize: Whether to normalize the remaining coordinate.
        :param colors: ((R, G, B), (R, G, B), (R, G, B)). Colors for front, side & rear.
        :param linewidth: Linewidth for plot.
        """
        corners = view_points(self.corners(), view, normalize=normalize)[:2, :]

        def draw_rect(selected_corners, color):
            prev = selected_corners[-1]
            for corner in selected_corners:
                cv2.line(im,
                         (int(prev[0]), int(prev[1])),
                         (int(corner[0]), int(corner[1])),
                         color, linewidth)
                prev = corner

        # Draw the sides
        for i in range(4):
            cv2.line(im,
                     (int(corners.T[i][0]), int(corners.T[i][1])),
                     (int(corners.T[i + 4][0]), int(corners.T[i + 4][1])),
                     colors[2][::-1], linewidth)

        # Draw front (first 4 corners) and rear (last 4 corners) rectangles(3d)/lines(2d)
        draw_rect(corners.T[:4], colors[0][::-1])
        draw_rect(corners.T[4:], colors[1][::-1])

        # Draw line indicating the front
        center_bottom_forward = np.mean(corners.T[2:4], axis=0)
        center_bottom = np.mean(corners.T[[2, 3, 7, 6]], axis=0)
        cv2.line(im,
                 (int(center_bottom[0]), int(center_bottom[1])),
                 (int(center_bottom_forward[0]), int(center_bottom_forward[1])),
                 colors[0][::-1], linewidth)

    def copy(self) -> 'Box':
        """
        Create a copy of self.
        :return: A copy.
        """
        return copy.deepcopy(self)