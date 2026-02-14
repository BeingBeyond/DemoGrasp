from isaacgymenvs.utils.torch_jit_utils import *
import torch
import os
import numpy as np
#import trimesh
#import open3d as o3d
#import queue
#from queue import Empty

def batch_linear_interpolate_poses(
    pose1: torch.Tensor,  # Shape: [B, 7] (x, y, z, qx, qy, qz, qw)
    pose2: torch.Tensor,  # Shape: [B, 7]
    max_trans_step: float,
    max_rot_step: float,
):
    """Batch interpolate between poses with limits on translation/rotation steps.
    
    Args:
        pose1: Starting poses of shape [B, 7]
        pose2: Target poses of shape [B, 7]
        max_trans_step: Maximum translation step between consecutive poses
        max_rot_step: Maximum rotation step (in radians) between consecutive poses
        
    Returns:
        interp_poses: Interpolated poses of shape [B, T_max, 7]
        timesteps: Actual lengths of each sequence in the batch [B]
    """
    B = pose1.shape[0]
    device = pose1.device
    
    # Split into positions and quaternions
    p1, q1 = pose1[:, :3], pose1[:, 3:]  # [B, 3], [B, 4]
    p2, q2 = pose2[:, :3], pose2[:, 3:]  # [B, 3], [B, 4]
    
    # --- Compute required steps for each pair ---
    delta_p = p2 - p1  # [B, 3]
    trans_dist = torch.norm(delta_p, dim=1)  # [B]
    n_trans = torch.ceil(trans_dist / max_trans_step).long().clamp(min=1)  # [B]
    
    theta = quat_diff_rad(q1, q2)  # [B]
    n_rot = torch.ceil(theta / max_rot_step).long().clamp(min=1)  # [B]
    
    # print(n_trans, n_rot)
    n = torch.maximum(n_trans, n_rot)  # [B]
    T_max = n.max().item()
    timesteps = n  # [B]
    
    # Create mask for valid steps [B, T_max+1]
    step_idx = torch.arange(T_max + 1, device=device).expand(B, -1)  # [B, T_max+1]
    valid_mask = step_idx <= n.unsqueeze(1)  # [B, T_max+1]
    
    # Compute interpolation factors t [B, T_max+1]
    t = step_idx.float() / n.unsqueeze(1).clamp(min=1)  # [B, T_max+1]
    t = t * valid_mask.float()  # Zero out invalid steps
    
    # Interpolate positions (LERP) [B, T_max+1, 3]
    interp_p = p1.unsqueeze(1) + t.unsqueeze(-1) * delta_p.unsqueeze(1)
    
    # --- Vectorized SLERP implementation ---
    interp_q = slerp(
        q1.unsqueeze(1).repeat(1,T_max+1,1), 
        q2.unsqueeze(1).repeat(1,T_max+1,1),
        t.unsqueeze(-1)
    ) # [B, T_max+1, 4]

    # Combine into poses [B, T_max+1, 7]
    interp_poses = torch.cat([interp_p, interp_q], dim=-1)
    
    return interp_poses, timesteps


COLORS_DICT = {
    "red": [1.0, 0.0, 0.0],
    "green": [0.0, 1.0, 0.0],
    "blue": [0.0, 0.0, 1.0],
    
    "yellow": [1.0, 1.0, 0.0],
    "cyan": [0.0, 1.0, 1.0],
    "magenta": [1.0, 0.0, 1.0],
    
    "white": [1.0, 1.0, 1.0],
    "black": [0.0, 0.0, 0.0],
    "gray": [0.5, 0.5, 0.5],
    "light_gray": [0.75, 0.75, 0.75],
    "dark_gray": [0.25, 0.25, 0.25],
    
    "orange": [1.0, 0.65, 0.0],
    "purple": [0.5, 0.0, 0.5],
    "pink": [1.0, 0.75, 0.8],
    "brown": [0.65, 0.16, 0.16],
    "olive": [0.5, 0.5, 0.0],
    "teal": [0.0, 0.5, 0.5],
    "navy": [0.0, 0.0, 0.5],
    "maroon": [0.5, 0.0, 0.0],
    "lime": [0.75, 1.0, 0.0],
    
    "gold": [1.0, 0.84, 0.0],
    "silver": [0.75, 0.75, 0.75],
    "bronze": [0.8, 0.5, 0.2],
    
    "sky_blue": [0.53, 0.81, 0.92],
    "forest_green": [0.13, 0.55, 0.13],
    "violet": [0.93, 0.51, 0.93],
    "coral": [1.0, 0.5, 0.31],
    "salmon": [0.98, 0.5, 0.45],
    "turquoise": [0.25, 0.88, 0.82],
    "indigo": [0.29, 0.0, 0.51],
    "beige": [0.96, 0.96, 0.86],
    "ivory": [1.0, 1.0, 0.94]
}


#################### point cloud for isaac
def load_object_point_clouds(object_files, asset_root):
    ret = []
    for fn in object_files:
        substrs = fn.split('/')
        assert len(substrs)==3, f"Filename should be ObjDatasetName/urdf/xxx.urdf, got {fn}"
        pc_fn = os.path.join(substrs[0], 'pointclouds', substrs[-1].replace('.urdf','.npy'))
        print("object file: {} -> pcl file: {}".format(fn, pc_fn))
        pc = np.load(os.path.join(asset_root, pc_fn))
        #pc = np.load("vision/real_pcl.npy")
        ret.append(pc)
    return ret

@torch.jit.script
def transform_points(quat, pt_input):
    quat_con = quat_conjugate(quat)
    pt_new = quat_mul(quat_mul(quat, pt_input), quat_con)
    if len(pt_new.size()) == 3:
        return pt_new[:,:,:3]
    elif len(pt_new.size()) == 2:
        return pt_new[:,:3]


def farthest_point_sample(xyz, npoint, device, init=None):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    B, N, C = xyz.size()
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    if init is not None:
        farthest = torch.tensor(init).long().reshape(B).to(device)
    else:
        farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, C)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def index_points(points, idx, device):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    B = points.size()[0]
    view_shape = list(idx.size())
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.size())
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

