import torch
import numpy as np


def batch_transform_trajs_to_local_frame(trajs, ref_idx=-1):
    """
    Batch transform trajectories to the local frame of reference.

    Args:
        trajs (torch.Tensor): Trajectories tensor of shape [B, N, T, x].
        ref_idx (int): Reference index for the local frame. Default is -1.

    Returns:
        torch.Tensor: Transformed trajectories in the local frame.

    """
    x = trajs[..., 0]
    y = trajs[..., 1]
    theta = trajs[..., 2]
    v_x = trajs[..., 3]
    v_y = trajs[..., 4]
    
    local_x = (x - x[:, :, ref_idx, None]) * torch.cos(theta[:, :, ref_idx, None]) + \
        (y - y[:, :, ref_idx, None]) * torch.sin(theta[:, :, ref_idx, None])
    local_y = -(x - x[:, :, ref_idx, None]) * torch.sin(theta[:, :, ref_idx, None]) + \
        (y - y[:, :, ref_idx, None]) * torch.cos(theta[:, :, ref_idx, None])
    
    local_theta = theta - theta[:, :, ref_idx, None]
    local_theta = wrap_angle(local_theta)

    local_v_x = v_x * torch.cos(theta[:, :, ref_idx, None]) + v_y * torch.sin(theta[:, :, ref_idx, None])
    local_v_y = -v_x * torch.sin(theta[:, :, ref_idx, None]) + v_y * torch.cos(theta[:, :, ref_idx, None])

    local_trajs = torch.stack([local_x, local_y, local_theta, local_v_x, local_v_y], dim=-1)
    local_trajs[trajs[..., :5] == 0] = 0

    if trajs.shape[-1] > 5:
        trajs = torch.cat([local_trajs, trajs[..., 5:]], dim=-1)
    else:
        trajs = local_trajs

    return trajs


def batch_transform_polylines_to_local_frame(polylines):
    """
    Batch transform polylines to the local frame of reference.

    Args:
        polylines (torch.Tensor): Polylines tensor of shape [B, M, W, 5].

    Returns:
        torch.Tensor: Transformed polylines in the local frame.

    """
    x = polylines[..., 0]
    y = polylines[..., 1]
    theta = polylines[..., 2]
    
    local_x = (x - x[:, :, 0, None]) * torch.cos(theta[:, :, 0, None]) + \
        (y - y[:, :, 0, None]) * torch.sin(theta[:, :, 0, None])
    local_y = -(x - x[:, :, 0, None]) * torch.sin(theta[:, :, 0, None]) + \
        (y - y[:, :, 0, None]) * torch.cos(theta[:, :, 0, None])
    
    local_theta = theta - theta[:, :, 0, None]
    local_theta = wrap_angle(local_theta)

    local_polylines = torch.stack([local_x, local_y, local_theta], dim=-1)
    local_polylines[polylines[..., :3] == 0] = 0
    polylines = torch.cat([local_polylines, polylines[..., 3:]], dim=-1)

    return polylines


def batch_transform_trajs_to_global_frame(trajs, current_states):
    """
    Batch transform trajectories to the global frame of reference.

    Args:
        trajs (torch.Tensor): Trajectories tensor of shape [B, N, x, 2 or 3].
        current_states (torch.Tensor): Current states tensor of shape [B, N, 5].

    Returns:
        torch.Tensor: Transformed trajectories in the global frame. [B, N, x, 3]

    """
    x, y, theta = current_states[:, :, 0], current_states[:, :, 1], current_states[:, :, 2]
    g_x = trajs[..., 0] * torch.cos(theta[:, :, None]) - trajs[..., 1] * torch.sin(theta[:, :,  None])
    g_y = trajs[..., 0] * torch.sin(theta[:, :, None]) + trajs[..., 1] * torch.cos(theta[:, :,  None])
    x = g_x + x[:, :, None]
    y = g_y + y[:, :, None]
        
    if trajs.shape[-1] == 2:
        trajs = torch.stack([x, y], dim=-1)
    else:
        theta = trajs[..., 2] + theta[:, :, None]
        theta = wrap_angle(theta)
        trajs = torch.stack([x, y, theta], dim=-1)

    return trajs


def wrap_angle(angle):
    """
    Wrap the angle to [-pi, pi].

    Args:
        angle (torch.Tensor): Angle tensor.

    Returns:
        torch.Tensor: Wrapped angle.

    """
    # return torch.atan2(torch.sin(angle), torch.cos(angle))
    return (angle + torch.pi) % (2 * torch.pi) - torch.pi


def inverse_kinematics(
    agents_future: torch.Tensor,
    agents_future_valid: torch.Tensor,
    dt: float = 0.1,
    action_len: int = 5,
):
    """
    Perform inverse kinematics to compute actions.

    Args:
        agents_future (torch.Tensor): Future agent positions tensor. 
            [B, A, T, 8] # x, y, yaw, velx, vely, length, width, height 
        agents_future_valid (torch.Tensor): Future agent validity tensor. [B, A, T]
        dt (float): Time interval. Default is 0.1.
        action_len (int): Length of each action. Default is 5.

    Returns:
        torch.Tensor: Predicted actions.

    """
    # Inverse kinematics implementation goes here
    batch_size, num_agents, num_timesteps, _ = agents_future.shape
    assert (num_timesteps-1) % action_len == 0, "future_len must be divisible by action_len"
    num_actions = (num_timesteps-1) // action_len
    
    yaw = agents_future[..., 2]
    speed = torch.norm(agents_future[..., 3:5], dim=-1)
    
    yaw_rate = wrap_angle(torch.diff(yaw, dim=-1)) / dt
    accel = torch.diff(speed, dim=-1) / dt
    action_valid = agents_future_valid[..., :1] & agents_future_valid[..., 1:]
    
    # filter out invalid actions
    yaw_rate = torch.where(action_valid, yaw_rate, 0.0)
    accel = torch.where(action_valid, accel, 0.0)
    
    # Reshape for mean pooling
    yaw_rate = yaw_rate.reshape(batch_size, num_agents, num_actions, -1)
    accel = accel.reshape(batch_size, num_agents, num_actions, -1)
    action_valid = action_valid.reshape(batch_size, num_agents, num_actions, -1)
    
    yaw_rate_sample = yaw_rate.sum(dim=-1) / torch.clamp(action_valid.sum(dim=-1), min=1.0)
    accel_sample = accel.sum(dim=-1) / torch.clamp(action_valid.sum(dim=-1), min=1.0)
    action = torch.stack([accel_sample, yaw_rate_sample], dim=-1)
    action_valid = action_valid.any(dim=-1)
    
    # Filter again
    action = torch.where(action_valid[..., None], action, 0.0)
    
    return action, action_valid


def roll_out(
        current_states: torch.Tensor,
        actions: torch.Tensor,
        dt: float = 0.1,
        action_len: int = 5,
        global_frame: float = True
    ):
        """
        Forward pass of the dynamics model.

        Args:
            current_states (torch.Tensor): Current states tensor of shape [B, N, x, 5]. [x, y, theta, v_x, v_y]
            actions (torch.Tensor): Inputs tensor of shape [B, N, x, T_f//T_a, 2]. [Accel, yaw_rate]
            global_frame (bool): Flag indicating whether to use the global frame of reference. Default is False.

        Returns:
            torch.Tensor: Predicted trajectories.

        """
        x = current_states[..., 0]
        y = current_states[..., 1]
        theta = current_states[..., 2]
        v_x = current_states[..., 3]
        v_y = current_states[..., 4]
        v = torch.sqrt(v_x**2 + v_y**2)

        a = actions[..., 0].repeat_interleave(action_len, dim=-1) 
        v = v.unsqueeze(-1) + torch.cumsum(a * dt, dim=-1) 
        v += torch.randn_like(v) * 0.1
        v = torch.clamp(v, min=0)

        yaw_rate = actions[..., 1].repeat_interleave(action_len, dim=-1) 
        yaw_rate += torch.randn_like(yaw_rate) * 0.01
    
        if global_frame:
            theta = theta.unsqueeze(-1) + torch.cumsum(yaw_rate * dt, dim=-1)
        else:
            theta = torch.cumsum(yaw_rate * dt, dim=2)
        # theta = torch.fmod(theta + torch.pi, 2*torch.pi) - torch.pi
        # theta = wrap_angle(theta)
        
        v_x = v * torch.cos(theta)
        v_y = v * torch.sin(theta)
        
        if global_frame:
            x = x.unsqueeze(-1) + torch.cumsum(v_x * dt, dim=-1)
            y = y.unsqueeze(-1) + torch.cumsum(v_y * dt, dim=-1)
        else:
            x = torch.cumsum(v_x * dt, dim=-1)
            y = torch.cumsum(v_y * dt, dim=-1)

        return torch.stack([x, y, theta, v_x, v_y], dim=-1)   


def construct_priors_simplex(num_labels, feature_len):
    """
    Construct a tensor of simplex priors (num_labels, feature_len)
    """
    basis = np.eye(num_labels)
    u = np.ones(num_labels)
    w = basis - u * (1. / (num_labels))
    scale = np.sqrt(float(num_labels) / float(num_labels - 1.))
    w_ = w * scale
    # w = np.concatenate([w, np.zeros([num_labels, feature_len - num_labels])], axis=-1)
    repeat = int(feature_len / num_labels)
    rest = feature_len - repeat * num_labels
    w = np.tile(w_, [1, repeat])
    if rest != 0:
        w = np.hstack([w, w_[:, :rest]])
    prior_means = torch.from_numpy(w).float()
    return prior_means


# below from LCSim

import math

import torch

EPS = 1e-10


def corners_from_bboxes(bboxes: torch.Tensor) -> torch.Tensor:
    assert bboxes.shape[-1] == 5  # [x, y, length, width, yaw]
    assert bboxes.ndim >= 2
    x, y, length, width, yaw = torch.unbind(bboxes, dim=-1)
    cos, sin = torch.cos(yaw), torch.sin(yaw)
    x_corners = torch.stack(
        [
            x + length / 2 * cos + width / 2 * sin,
            x - length / 2 * cos + width / 2 * sin,
            x - length / 2 * cos - width / 2 * sin,
            x + length / 2 * cos - width / 2 * sin,
        ],
        dim=-1,
    )
    y_corners = torch.stack(
        [
            y + length / 2 * sin - width / 2 * cos,
            y - length / 2 * sin - width / 2 * cos,
            y - length / 2 * sin + width / 2 * cos,
            y + length / 2 * sin + width / 2 * cos,
        ],
        dim=-1,
    )
    # [..., 4, 2]
    return torch.stack([x_corners, y_corners], dim=-1)


def has_overlap(bboxes_a: torch.Tensor, bboxes_b: torch.Tensor) -> torch.Tensor:
    """Checks if 5 dof bboxes (with any prefix shape) overlap with each other.

    It does a 1:1 comparison of equivalent batch indices.

    The algorithm first computes bboxes_a's projection on bboxes_b's axes and
    check if there is an overlap between the projection. It then computes
    bboxes_b's projection on bboxes_a's axes and check overlap. Two bboxes are
    overlapped if and only if there is overlap in both steps.

    Args:
      bboxes_a: Bounding boxes of the above format of shape (..., 5). The last
        dimension represents [x, y, length, width, yaw].
      bboxes_b: Bounding boxes of the above format of shape (..., 5).

    Returns:
      Boolean array which specifies whether `bboxes_a` and `bboxes_b` overlap each
        other of shape (...).
    """
    assert bboxes_a.shape[-1] == 5
    assert bboxes_b.shape == bboxes_a.shape

    def _overlap_a_over_b(first, second):
        c, s = torch.cos(first[..., 4]), torch.sin(first[..., 4])
        # [x, y, length, width, yaw]
        normals_t = torch.stack(
            [torch.stack([c, -s], axis=-1), torch.stack([s, c], axis=-1)],
            axis=-2,
        )
        corners_a = corners_from_bboxes(first)
        corners_b = corners_from_bboxes(second)
        proj_a = torch.matmul(corners_a, normals_t)
        min_a = torch.min(proj_a, axis=-2).values
        max_a = torch.max(proj_a, axis=-2).values
        proj_b = torch.matmul(corners_b, normals_t)
        min_b = torch.min(proj_b, axis=-2).values
        max_b = torch.max(proj_b, axis=-2).values
        distance = torch.minimum(max_a, max_b) - torch.maximum(min_a, min_b)
        return torch.all(distance > 0, axis=-1)

    return torch.logical_and(
        _overlap_a_over_b(bboxes_a, bboxes_b),
        _overlap_a_over_b(bboxes_b, bboxes_a),
    )


def compute_pairwise_overlaps(traj: torch.Tensor) -> torch.Tensor:
    """Computes an overlap mask among all agent pairs for all steps.

    5 dof trajectories have [x, y, length, width, yaw] for last dimension.

    Args:
      traj: Bounding boxes of the above format of shape (num_objects, 5).

    Returns:
      Boolean array of shape (num_objects, ) which denotes whether
        any of the objects in the trajectory are in overlap.
    """
    assert traj.shape[-1] == 5
    assert traj.ndim == 2

    # (num_objects, num_objects, 5)
    traj_a = traj.unsqueeze(0).expand(traj.shape[0], -1, -1)
    traj_b = traj.unsqueeze(1).expand(-1, traj.shape[0], -1)
    self_mask = torch.eye(traj.shape[0], dtype=torch.bool, device=traj.device)
    # (num_objects, num_objects)
    return torch.where(self_mask, False, has_overlap(traj_a, traj_b)).sum(dim=-1) > 0


# def compute_overlap_rate(
#     data: HeteroData, traj: torch.Tensor, mask: torch.Tensor
# ) -> torch.Tensor:
#     """compute the overlap rate between agents of predicted trajectory

#     Args:
#         data (HeteroData): batch data
#         traj (torch.Tensor): predicted trajectory of shape (num_agents, num_steps, 5).
#             The last dimension represents [x, y, length, width, yaw].
#         mask (torch.Tensor): mask of shape (num_agents, ) indicating valid agents.

#     Returns:
#         torch.Tensor: overlap rate of shape (1, )
#     """
#     assert traj.shape[-1] == 5
#     assert traj.ndim == 3
#     assert traj.shape[0] == data["agent"]["xyz"].shape[0]

#     batch = (
#         data["agent"]["batch"] if isinstance(data, Batch) else torch.zeros_like(mask)
#     )
#     batch_size = batch.max().item() + 1
#     # traj = traj[mask]
#     overlap_cnt = 0
#     for i in range(traj.shape[1]):
#         # for b in range(batch_size):
#         #     mask_b = batch == b
#         #     if not torch.any(mask_b & mask):
#         #         continue
#         #     traj_b = traj[mask_b, i]
#         #     mask_b = mask[mask_b]
#         #     overlap = compute_pairwise_overlaps(traj_b)
#         #     overlap_cnt += torch.sum(overlap & mask_b)
#         overlap = compute_pairwise_overlaps(traj[:, i])
#         overlap_cnt += torch.sum(overlap & mask)
#     overlap_rate = overlap_cnt / torch.sum(mask) / traj.shape[1]
#     return overlap_rate


