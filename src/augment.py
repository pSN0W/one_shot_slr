import random
import os

import numpy as np

MAX_SHIFT = 0.1
MIN_SHIFT = 0.4

MIN_SCALE = 0.7
MAX_SCALE = 1.3

MIN_FRAME = 2
MAX_FRAME = 10


def get_pose_and_other_landmark(arr):
    pose = arr[:, : 33 * 4].reshape(-1, 33, 4)
    lh = arr[:, 33 * 4 : 33 * 4 + 21 * 3].reshape(-1, 21, 3)
    rh = arr[:, 33 * 4 + 21 * 3 :].reshape(-1, 21, 3)
    return pose, lh, rh


def shift_along_time(data):
    data = data.copy()
    num_frames_to_move = random.randint(MIN_FRAME, MAX_FRAME)
    return np.roll(data, (num_frames_to_move + 1) * random.choice([1, -1]))


def flip(arr):
    arr = arr.copy()
    pose, lh, rh = get_pose_and_other_landmark(arr)
    pose[:, :, 0] = 1 - pose[:, :, 0]
    rh[:, :, 0] = 1 - rh[:, :, 0]
    lh[:, :, 0] = 1 - lh[:, :, 0]
    return np.hstack(
        [
            pose.reshape(arr.shape[0], -1),
            lh.reshape(arr.shape[0], -1),
            rh.reshape(arr.shape[0], -1),
        ]
    )


def shift_to_right_left(arr):
    shift = random.uniform(MAX_SHIFT, MIN_SHIFT) * random.choice([1, -1])
    arr = arr.copy()
    pose, lh, rh = get_pose_and_other_landmark(arr)
    pose[:, :, 0] = shift + pose[:, :, 0]
    rh[:, :, 0] = shift + rh[:, :, 0]
    lh[:, :, 0] = shift + lh[:, :, 0]
    return np.hstack(
        [
            pose.reshape(arr.shape[0], -1),
            lh.reshape(arr.shape[0], -1),
            rh.reshape(arr.shape[0], -1),
        ]
    )


def scale_image(arr):
    scale = random.uniform(MAX_SCALE, MIN_SCALE)
    axis = random.choice([0, 1])
    arr = arr.copy()
    pose, lh, rh = get_pose_and_other_landmark(arr)
    pose[:, :, axis] = scale * pose[:, :, axis]
    rh[:, :, axis] = scale * rh[:, :, axis]
    lh[:, :, axis] = scale * lh[:, :, axis]
    return np.hstack(
        [
            pose.reshape(arr.shape[0], -1),
            lh.reshape(arr.shape[0], -1),
            rh.reshape(arr.shape[0], -1),
        ]
    )


def rotate_7_degree(pose, way):
    theta = 7 * way * (np.pi / 180)
    c, s = np.cos(theta), np.sin(theta)
    rotation_matrix = np.array([[c, -s], [s, c]])

    rotated = pose @ rotation_matrix

    return rotated


def rotate_image(arr):
    way = random.choice([1, -1])

    arr = arr.copy()
    pose, lh, rh = get_pose_and_other_landmark(arr)
    pose[:, :, :2] = rotate_7_degree(pose[:, :, :2], way)
    rh[:, :, :2] = rotate_7_degree(rh[:, :, :2], way)
    lh[:, :, :2] = rotate_7_degree(lh[:, :, :2], way)
    return np.hstack(
        [
            pose.reshape(arr.shape[0], -1),
            lh.reshape(arr.shape[0], -1),
            rh.reshape(arr.shape[0], -1),
        ]
    )


AUGMENT_FNS = [shift_along_time, flip, shift_to_right_left, scale_image, rotate_image]
FILE_SUFFIXES = ["shifted_by_time", "flipped", "shifted", "scaled", "rotated"]


def augment(src, dest):
    data = np.load(src)
    np.save(dest, data)
    for fn, suffix in zip(AUGMENT_FNS, FILE_SUFFIXES):
        file_name = f"{dest[:-4]}_{suffix}.npy"
        transformed_data = fn(data)
        np.save(file_name, transformed_data)


def augment_dir(source_dir, destination_dir):
    for root, dirs, files in os.walk(source_dir):
        # Create corresponding directories in the destination directory
        for dir_name in dirs:
            source_path = os.path.join(root, dir_name)
            relative_path = os.path.relpath(source_path, source_dir)
            destination_path = os.path.join(destination_dir, relative_path)

            if not os.path.exists(destination_path):
                os.makedirs(destination_path)

        # Copy files from the source directory to the destination directory
        for file_name in files:
            source_path = os.path.join(root, file_name)
            relative_path = os.path.relpath(source_path, source_dir)
            destination_path = os.path.join(destination_dir, relative_path)
            destination_path = os.path.splitext(destination_path)[0] + ".npy"
            if not os.path.exists(destination_path):
                print("Processing : ", source_path)
                augment(source_path, destination_path)


SOURCE_DIR = "/home/sn0w/Desktop/IIITA/Sem7/mini/data/include_same_length_keypoint"
DEST_DIR = "/home/sn0w/Desktop/IIITA/Sem7/mini/data/include_same_length_keypoint_augmented"

augment_dir(
    source_dir=SOURCE_DIR,
    destination_dir=DEST_DIR
)
