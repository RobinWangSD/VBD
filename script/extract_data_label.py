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
import glob
import argparse
import pickle
from vbd.data.data_utils import *
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map  # or thread_map

from waymax import dataloader
from waymax.config import DataFormat
import functools


def extract_high_level_motion_action(heading, acceleration):
    if acceleration > 1: 
        speed_action = 1 # acceleration
    elif acceleration < -1:
        speed_action = 2 # deceleration
    else:
        speed_action = 3 # keep speed
    
    heading = np.rad2deg(heading)
    if np.abs(heading) < 2.4:
        steering_action = 0 # go straight
    elif np.abs(heading) < 26.4: 
        if heading > 0 :
            steering_action = 1 # turn left
        else: 
            steering_action = 2 # turn right
    else:
        if heading > 0 :
            steering_action = 3 # left u turn
        else: 
            steering_action = 2 # turn right
    
    return np.array([speed_action, steering_action])

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


def extract_patch_action(speed_patch, heading_patch):
    ## no need for sdc
    # first_valid_ts = -1
    # last_valid_ts = -1
    # for ts in range(valid_patch.shape[0]):
    #     if first_valid_ts==-1 and valid_patch[ts]:
    #         first_valid_ts = ts
    #     elif first_valid_ts!=-1 and last_valid_ts==-1:
    #         if not valid_patch[ts]:
    #             last_valid_ts = ts - 1
    #         elif ts == valid_patch.shape[0]-1:
    #             last_valid_ts = ts
    # if first_valid_ts==-1 and last_valid_ts==-1 or first_valid_ts==last_valid_ts:
    #     return np.array([-1, -1], dtype=np.float32)
    assert len(speed_patch) == len(heading_patch)
    speed_diff = 10 * (speed_patch[-1] - speed_patch[0]) / len(speed_patch)
    heading_diff = 10 * wrap_angle(heading_patch[-1] - heading_patch[0]) / len(heading_patch)
    heading_diff = wrap_angle(heading_diff)
    patch_action = extract_high_level_motion_action(heading_diff, speed_diff)
    return patch_action


def extract_patches_action(speed, heading, sample_rate=10):
    high_level_action = []
    for patch_id in range((speed.shape[0]) // sample_rate):
        speed_patch = speed[patch_id*sample_rate:(patch_id+1)*sample_rate]
        heading_patch = heading[patch_id*sample_rate:(patch_id+1)*sample_rate]
        high_level_action.append(extract_patch_action(speed_patch, heading_patch))
    return np.stack(high_level_action, axis=0)


def extract_sdc_action(data):
    scenario = data['scenario_raw']
    sdc_id = np.where(scenario.object_metadata.is_sdc)[0][0]
    sdc_id_in_processed = np.where(data["agents_id"]==sdc_id)[0][0]
    sdc_future = data["agents_future"][sdc_id_in_processed]
    assert sdc_future.shape[0] == 81 and sdc_future.shape[1] == 5, "sdc future traj shape is wrong"
    vel_xy = sdc_future[:, 3:]
    speed = np.linalg.norm(vel_xy, axis=-1)
    heading = sdc_future[:, 2]
    sdc_future_actions_4s = extract_patches_action(speed, heading, sample_rate=40)
    sdc_future_actions_1s = extract_patches_action(speed, heading, sample_rate=10)
    return sdc_id, sdc_future_actions_4s, sdc_future_actions_1s


if __name__ == '__main__':
    dataset_type = sys.argv[1]  # training / validation
    with open(f"/robin-west/VBD/script/{dataset_type}_file_buckets.pkl", "rb") as buckets_f:
        buckets = pickle.load(buckets_f)
    bucket_id = int(sys.argv[2])
    data_file_list = buckets[bucket_id]

    action_labels = dict() 
    for data_file_path in tqdm(data_file_list):
        scenario_id = data_file_path.split("/")[-1].rstrip(".pkl").split("_")[-1]
        with open(data_file_path, "rb") as data_f:
            data = pickle.load(data_f)
        sdc_id, sdc_future_actions_4s, sdc_future_actions_1s = extract_sdc_action(data)
        action_labels[scenario_id] = {
            'sdc_id': sdc_id,
            '1s_action': sdc_future_actions_1s,
            '4s_action': sdc_future_actions_4s,
        }
    with open(f"/robin-west/womd_processed/vbd/{dataset_type}/action_labels_bucket_{bucket_id}.pkl", "wb") as action_labels_f:
        pickle.dump(action_labels, action_labels_f)