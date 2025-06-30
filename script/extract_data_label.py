import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# disable GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
# set tf to cpu only
tf.config.set_visible_devices([], 'GPU')
import jax
jax.config.update('jax_platform_name', 'cpu')

import sys
sys.path.append("/robin-west/VBD")

import sys
import glob
import argparse
import pickle
from vbd.data.data_utils import *
from vbd.model.model_utils import inverse_kinematics
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map  # or thread_map

from waymax import dataloader
from waymax.config import DataFormat
import functools

import torch


# def extract_high_level_motion_action(heading, acceleration):
#     if acceleration > 2: 
#         speed_action = 1 # acceleration
#     elif acceleration < -2:
#         speed_action = 2 # deceleration
#     else:
#         speed_action = 3 # keep speed
    
#     heading = np.rad2deg(heading)
#     if np.abs(heading) < 6:
#         steering_action = 0 # go straight
#     elif np.abs(heading) < 30: 
#         if heading > 0 :
#             steering_action = 1 # turn left
#         else: 
#             steering_action = 2 # turn right
#     else:
#         if heading > 0 :
#             steering_action = 3 # left u turn
#         else: 
#             steering_action = 2 # turn right
    
#     return np.array([speed_action, steering_action])

# def wrap_angle(angle):
#     """
#     Wrap the angle to [-pi, pi].

#     Args:
#         angle (torch.Tensor): Angle tensor.

#     Returns:
#         torch.Tensor: Wrapped angle.

#     """
#     # return torch.atan2(torch.sin(angle), torch.cos(angle))
#     return (angle + torch.pi) % (2 * torch.pi) - torch.pi


# def extract_patch_action(speed_patch, heading_patch):
#     ## no need for sdc
#     # first_valid_ts = -1
#     # last_valid_ts = -1
#     # for ts in range(valid_patch.shape[0]):
#     #     if first_valid_ts==-1 and valid_patch[ts]:
#     #         first_valid_ts = ts
#     #     elif first_valid_ts!=-1 and last_valid_ts==-1:
#     #         if not valid_patch[ts]:
#     #             last_valid_ts = ts - 1
#     #         elif ts == valid_patch.shape[0]-1:
#     #             last_valid_ts = ts
#     # if first_valid_ts==-1 and last_valid_ts==-1 or first_valid_ts==last_valid_ts:
#     #     return np.array([-1, -1], dtype=np.float32)
#     assert len(speed_patch) == len(heading_patch)
#     speed_diff = 10 * (speed_patch[-1] - speed_patch[0]) / len(speed_patch)
#     heading_diff = 10 * wrap_angle(heading_patch[-1] - heading_patch[0]) / len(heading_patch)
#     heading_diff = wrap_angle(heading_diff)
#     patch_action = extract_high_level_motion_action(heading_diff, speed_diff)
#     return patch_action


# def extract_patches_action(speed, heading, sample_rate=10):
#     high_level_action = []
#     for patch_id in range((speed.shape[0]) // sample_rate):
#         speed_patch = speed[patch_id*sample_rate:(patch_id+1)*sample_rate]
#         heading_patch = heading[patch_id*sample_rate:(patch_id+1)*sample_rate]
#         high_level_action.append(extract_patch_action(speed_patch, heading_patch))
#     return np.stack(high_level_action, axis=0)


# def extract_sdc_action(data):
#     scenario = data['scenario_raw']
#     sdc_id = np.where(scenario.object_metadata.is_sdc)[0][0]
#     sdc_id_in_processed = np.where(data["agents_id"]==sdc_id)[0][0]
#     sdc_future = data["agents_future"][sdc_id_in_processed]
#     assert sdc_future.shape[0] == 81 and sdc_future.shape[1] == 5, "sdc future traj shape is wrong"
#     vel_xy = sdc_future[:, 3:]
#     speed = np.linalg.norm(vel_xy, axis=-1)
#     heading = sdc_future[:, 2]
#     # sdc_future_actions_4s = extract_patches_action(speed, heading, sample_rate=40)
#     sdc_future_actions_1s = extract_patches_action(speed, heading, sample_rate=10)
#     return sdc_id, sdc_future_actions_1s

class TrajectoryType:
    STATIONARY = 0
    STRAIGHT = 1
    STRAIGHT_RIGHT = 2
    STRAIGHT_LEFT = 3
    RIGHT_U_TURN = 4
    RIGHT_TURN = 5
    LEFT_U_TURN = 6
    LEFT_TURN = 7


def classify_track(start_point, end_point, start_velocity, end_velocity, start_heading, end_heading):
    # The classification strategy is taken from
    # waymo_open_dataset/metrics/motion_metrics_utils.cc#L28

    # Parameters for classification, taken from WOD
    kMaxSpeedForStationary = 2.0  # (m/s)
    kMaxDisplacementForStationary = 5.0  # (m)
    kMaxLateralDisplacementForStraight = 5.0  # (m)
    kMinLongitudinalDisplacementForUTurn = -5.0  # (m)
    kMaxAbsHeadingDiffForStraight = np.pi / 6.0  # (rad)

    x_delta = end_point[0] - start_point[0]
    y_delta = end_point[1] - start_point[1]

    final_displacement = np.hypot(x_delta, y_delta)
    heading_diff = end_heading - start_heading
    normalized_delta = np.array([x_delta, y_delta])
    rotation_matrix = np.array([[np.cos(-start_heading), -np.sin(-start_heading)],
                                [np.sin(-start_heading), np.cos(-start_heading)]])
    normalized_delta = np.dot(rotation_matrix, normalized_delta)
    start_speed = np.hypot(start_velocity[0], start_velocity[1])
    end_speed = np.hypot(end_velocity[0], end_velocity[1])
    max_speed = max(start_speed, end_speed)
    dx, dy = normalized_delta

    # Check for different trajectory types based on the computed parameters.
    if max_speed < kMaxSpeedForStationary and final_displacement < kMaxDisplacementForStationary:
        return TrajectoryType.STATIONARY
    if np.abs(heading_diff) < kMaxAbsHeadingDiffForStraight:
        if np.abs(normalized_delta[1]) < kMaxLateralDisplacementForStraight:
            return TrajectoryType.STRAIGHT
        return TrajectoryType.STRAIGHT_RIGHT if dy < 0 else TrajectoryType.STRAIGHT_LEFT
    if heading_diff < -kMaxAbsHeadingDiffForStraight and dy < 0:
        return TrajectoryType.RIGHT_U_TURN if normalized_delta[
                                                  0] < kMinLongitudinalDisplacementForUTurn else TrajectoryType.RIGHT_TURN
    if dx < kMinLongitudinalDisplacementForUTurn:
        return TrajectoryType.LEFT_U_TURN
    return TrajectoryType.LEFT_TURN


def extract_sdc_action(data):
    scenario = data['scenario_raw']
    sdc_id = np.where(scenario.object_metadata.is_sdc)[0][0]
    sdc_id_in_processed = np.where(data["agents_id"]==sdc_id)[0][0]
    sdc_future = data["agents_future"][sdc_id_in_processed]
    assert sdc_future.shape[0] == 81 and sdc_future.shape[1] == 5, "sdc future traj shape is wrong"

    start_point = sdc_future[0, 0:2]
    end_point = sdc_future[-1, 0:2]
    start_velocity = sdc_future[0, 3:]
    end_velocity = sdc_future[-1, 3:]
    start_heading = sdc_future[0, 2]
    end_heading = sdc_future[-1, 2]

    sdc_future_valid = torch.from_numpy(np.ones(sdc_future.shape[:1], dtype=bool)[None, None, :])
    sdc_future = torch.from_numpy(sdc_future[None, None, :, :])
    sdc_actions, sdc_action_valid = inverse_kinematics(sdc_future, sdc_future_valid, action_len=2)
    action_mean = np.array([0.0, 0.0])
    action_std = np.array([1.0, 0.15])
    sdc_actions_normalized = (sdc_actions - action_mean) / action_std

    action = classify_track(start_point, end_point, start_velocity, end_velocity, start_heading, end_heading)
    return sdc_id, action, sdc_actions_normalized



# def action_1s_to_8s(action_1s, steer_thres = 2, speed_thres = 2):
#     steer_action = action_1s[:, 1]
#     speed_action = action_1s[:, 0]

#     steer_dict = {}
#     speed_dict = {}

#     for steer in steer_action:
#         steer_dict[steer] = steer_dict.get(steer, 0) + 1

#     for speed in speed_action:
#         speed_dict[speed] = speed_dict.get(speed, 0) + 1

#     # parse steer
#     steer_8s = 0
#     steer_label_cnt = -1
#     for key in [1, 2, 3]:
#         if key != 0 and steer_dict.get(key, 0)>=steer_thres and steer_dict.get(key, 0) > steer_label_cnt:
#             steer_8s = key
#             steer_label_cnt = steer_dict.get(key, 0)


#     # parse speed
#     speed_8s = 3
#     speed_label_cnt = -1
#     for key in [1, 2]:
#         if key != 3 and speed_dict.get(key,0)>=speed_thres and speed_dict.get(key,0) > speed_label_cnt:
#             speed_8s = key
#             speed_label_cnt = speed_dict.get(key,0)
#     return np.array([speed_8s, steer_8s])


# def action_1s_to_4s(action_1s, steer_thres = 2, speed_thres = 2):
#     labels_4s = np.zeros((2,2))
#     for i in range(2):
#         steer_action = action_1s[4*i:4*(i+1), 1]
#         speed_action = action_1s[4*i:4*(i+1), 0]

#         steer_dict = {}
#         speed_dict = {}

#         for steer in steer_action:
#             steer_dict[steer] = steer_dict.get(steer, 0) + 1

#         for speed in speed_action:
#             speed_dict[speed] = speed_dict.get(speed, 0) + 1

#         # parse steer
#         steer_4s = 0
#         steer_label_cnt = -1

#         for key in [1, 2, 3]:
#             if key != 0 and steer_dict.get(key, 0) >= steer_thres and steer_dict.get(key, 0) >= steer_label_cnt:
#                 steer_4s = key
#                 steer_label_cnt = steer_dict.get(key, 0)


#         # parse speed
#         speed_4s = 3
#         speed_label_cnt = -1
#         for key in [1,2]:
#             if key != 3 and speed_dict.get(key,0) >= speed_thres and speed_dict.get(key,0) >= speed_label_cnt:
#                 speed_4s = key
#                 speed_label_cnt = speed_dict[key]
#         labels_4s[i,0] = speed_4s
#         labels_4s[i,1] = steer_4s
#     return labels_4s


# def parse_action_1s_to_8s(scenario_id_list, action_labels, steer_thres = 2, speed_thres = 2):
#     action_labels_8s_dict = {}
#     for scenario_id in scenario_id_list:
#         action_labels_1s = action_labels[scenario_id]['1s_action']
#         action_labels_8s = action_1s_to_8s(action_labels_1s, steer_thres = 2, speed_thres = 2)
#         action_labels_8s_dict[scenario_id] = action_labels_8s_dict.get(scenario_id, {})
#         action_labels_8s_dict[scenario_id]['8s_action'] = action_labels_8s
#         action_labels_8s_dict[scenario_id]['1s_action'] = action_labels_1s
#         action_labels_8s_dict[scenario_id]['4s_action'] = action_labels[scenario_id]['4s_action']
#         action_labels_8s_dict[scenario_id]['sdc_id'] = action_labels[scenario_id]['sdc_id']
#     return action_labels_8s_dict



if __name__ == '__main__':
    print('getting proccess list')
    dataset_type = sys.argv[1]  # training / validation
    with open(f"/robin-west/VBD/script/{dataset_type}_file_buckets.pkl", "rb") as buckets_f:
        buckets = pickle.load(buckets_f)
    bucket_id = int(sys.argv[2])
    print(f'processing bucket id {bucket_id}')
    data_file_list = buckets[bucket_id]

    action_labels = dict()
    print('start processing') 
    for data_file_path in tqdm(data_file_list):
        scenario_id = data_file_path.split("/")[-1].rstrip(".pkl").split("_")[-1]
        # print(f"{scenario_id} processed")
        with open(data_file_path, "rb") as data_f:
            data = pickle.load(data_f)
        # sdc_id, sdc_future_actions_1s = extract_sdc_action(data)
        # sdc_future_actions_4s = action_1s_to_4s(sdc_future_actions_1s, steer_thres = 2, speed_thres = 2)
        # sdc_future_actions_8s = action_1s_to_8s(sdc_future_actions_1s, steer_thres = 2, speed_thres = 2)
        sdc_id, action_label, normalized_action = extract_sdc_action(data)
        action_labels[scenario_id] = {
            'sdc_id': sdc_id,
            'action_label': action_label,
            'normalized_action': normalized_action.numpy(),
            # '4s_action': sdc_future_actions_4s,
            # '8s_action': sdc_future_actions_8s,
        }
    with open(f"/robin-west/womd_processed/vbd/{dataset_type}/action_labels/action_labels_with_normalized_action_bucket_{bucket_id}.pkl", "wb") as action_labels_f:
        pickle.dump(action_labels, action_labels_f)