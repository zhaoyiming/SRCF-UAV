import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # 后端改为非交互式

def unnormalize_query_bbox(query_bbox_norm: np.ndarray, point_cloud_range: list) -> np.ndarray:
    """
    Restore normalized query_bbox values to real-world units.
    """
    x_min, y_min, z_min, x_max, y_max, z_max = point_cloud_range
    range_x = x_max - x_min
    range_y = y_max - y_min
    range_z = z_max - z_min

    x = query_bbox_norm[:, 0] * range_x + x_min
    y = query_bbox_norm[:, 1] * range_y + y_min
    z = query_bbox_norm[:, 2] * range_z + z_min

    # other columns (w, l, h, sin_yaw, cos_yaw, vx, vy) are unchanged or unused here
    return np.stack([x, y, z], axis=1)

def plot_bev_and_height_points(query_bbox_norm: np.ndarray,
                                point_cloud_range: list,
                                lastfix: str = "ori",
                                output_dir: str = "debug_querybev"):
    """
    Plot BEV points and height points side by side.
    """
    os.makedirs(output_dir, exist_ok=True)
    points = unnormalize_query_bbox(query_bbox_norm, point_cloud_range)
    xs, ys, zs = points[:,0], points[:,1], points[:,2]

    fig, axs = plt.subplots(1, 2, figsize=(16, 8))
    # print(points.shape)

    # Left: BEV scatter
    axs[0].scatter(xs, ys, s=5)
    axs[0].set_xlim(point_cloud_range[0], point_cloud_range[3])
    axs[0].set_ylim(point_cloud_range[1], point_cloud_range[4])
    axs[0].set_xlabel('X (m)')
    axs[0].set_ylabel('Y (m)')
    axs[0].set_title('BEV Points of Queries')
    axs[0].grid(True)

    # Right: Height scatter
    axs[1].scatter(range(len(zs)), zs, s=5)
    axs[1].set_xlabel('Query Index')
    axs[1].set_ylabel('Height (m)')
    axs[1].set_title('Height of Queries')
    axs[1].grid(True)

    output_path = os.path.join(output_dir, f"bev_height_points_{lastfix}.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Scatter plots saved to {output_path}")

# Example usage
if __name__ == "__main__":
    query_bbox_norm = np.random.rand(900, 10)  # replace with actual normalized array
    point_cloud_range = [-25.6, -25.6, 0.0, 25.6, 25.6, 10]
    plot_bev_and_height_points(query_bbox_norm, point_cloud_range)
