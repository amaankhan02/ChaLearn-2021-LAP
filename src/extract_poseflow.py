'''Computes PoseFlows from pose keypoint files'''
import argparse
import glob
import json
import math
import os
from collections import defaultdict
import

import numpy as np

# reads the keypoint file 
def read_pose(kp_file):
    with open(kp_file) as kf:
        value = json.loads(kf.read())
        kps = value['people'][0]['pose_keypoints_2d']
        x = kps[0::3]
        y = kps[1::3]
        return np.stack((x, y), axis=1)

# TODO: THE ACTUAL CALCULATION BASED ON PREVIOUS FRAME AND CURRENT FRAME
def calc_pose_flow(prev, next):
    result = np.zeros_like(prev)
    for kpi in range(prev.shape[0]): # iterate thru the different landmarks --> shape[0] = # landmarks
        # kpi = keypoint index
        # if either frame's landmark value is just (x, y) = (0, 0)
        if np.count_nonzero(prev[kpi]) == 0 or np.count_nonzero(next[kpi]) == 0:
            # then just assign both to 0 and continue
            result[kpi, 0] = 0.0
            result[kpi, 1] = 0.0
            continue
        # atan2() - https://en.wikipedia.org/wiki/Atan2 --> arctan but has values for [0,2pi] instead of just [-pi,pi]
        ang = math.atan2(next[kpi, 1] - prev[kpi, 1], next[kpi, 0] - prev[kpi, 0])
        mag = np.linalg.norm(next[kpi] - prev[kpi])  # magnitude of the diff of the 2 vectors

        result[kpi, 0] = ang
        result[kpi, 1] = mag

    return result

# Pre-processing done before calculating pose flow
def impute_missing_keypoints(poses):
    """Replace missing keypoints (on the origin) by values from neighbouring frames."""
    # 1. Collect missing keypoints
    missing_keypoints = defaultdict(list)  # frame index -> keypoint indices that are missing
    for i in range(poses.shape[0]):
        for kpi in range(poses.shape[1]):
            if np.count_nonzero(poses[i, kpi]) == 0:  # Missing keypoint at (0, 0)
                missing_keypoints[i].append(kpi)
    # 2. Impute them
    for i in missing_keypoints.keys():
        missing = missing_keypoints[i]
        for kpi in missing:
            # Possible replacements
            candidates = poses[:, kpi]
            min_dist = np.inf
            replacement = -1
            for f in range(candidates.shape[0]):
                if f != i and np.count_nonzero(candidates[f]) > 0:
                    distance = abs(f - i)
                    if distance < min_dist:
                        min_dist = distance
                        replacement = f
            # Replace
            if replacement > -1:
                poses[i, kpi] = poses[replacement, kpi]
    # 3. We have imputed as many keypoints as possible with the closest non-missing temporal neighbours
    return poses

# so that different distances of person from camera shouldn't affect 
def normalize(poses):
    """Normalize each pose in the array to account for camera position. We normalize
    by dividing keypoints by a factor such that the length of the neck becomes 1."""
    # TODO: openpose is 2D Keypoints, but MediaPipe has 3D keypoints. so maybe i can just
    #       take into account the z coordinate, and not have to do this normalizing
    for i in range(poses.shape[0]):
        upper_neck = poses[i, 17]
        head_top = poses[i, 18]
        neck_length = np.linalg.norm(upper_neck - head_top)
        poses[i] /= neck_length
        assert math.isclose(np.linalg.norm(upper_neck - head_top), 1)
    return poses


def main(input_dir):
    input_dirs = sorted(glob.glob(os.path.join(input_dir, '*', '*_color.kp')))
    input_dir_index = 0
    total = len(input_dirs)  # number of videos

    for input_dir in input_dirs:  # iterate thru each folder, which represents each video
        print(f'{input_dir_index}/{total}')
        input_dir_index += 1

        output_dir = input_dir.replace('kp', 'kpflow2')
        os.makedirs(output_dir, exist_ok=True)

        # get the individual keypoint files from this video --> i.e, each file represents one frame
        kp_files = sorted(glob.glob(os.path.join(input_dir, '*.json')))

        # 1. Collect all keypoint files from dir and pre-process them
        poses = []
        for i in range(len(kp_files)):  # iterate thru all kp files
            poses.append(read_pose(kp_files[i]))
        poses = np.stack(poses)
        poses = impute_missing_keypoints(poses) # REVIEW THIS
        poses = normalize(poses)                # REVIEW THIS

        # import matplotlib.pyplot as plt
        # for i in range(poses.shape[0]):
        #     plt.figure()
        #     plt.scatter(poses[i, :, 0], -poses[i, :, 1])
        #     plt.show()
        # break

        # 2. Compute pose flow on pre-processed keypoints
        prev = poses[0]
        for i in range(1, poses.shape[0]):
            current_frame = poses[i]
            flow = calc_pose_flow(prev, current_frame)
            np.save(os.path.join(output_dir, 'flow_{:05d}'.format(i - 1)), flow)
            prev = current_frame


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--input_dir', type=str)
    # args = parser.parse_args()
    INPUT_DIR = r'C:\Users\amaan\GitHub\ChaLearn-2021-LAP\data\kp'
    main(INPUT_DIR)
