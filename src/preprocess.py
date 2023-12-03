import os
import random

import numpy as np


def interpolate_frames(video_embedding, required_frame_length):
    fin_array = np.empty(
        (required_frame_length, video_embedding.shape[1]), dtype="float"
    )
    additional_required_rows = required_frame_length - video_embedding.shape[0]
    idx_to_interpolate = set(
        random.sample(range(1, required_frame_length - 1), additional_required_rows)
    )

    assert (
        len(idx_to_interpolate) == additional_required_rows
    ), "Generated duplicate indexes"

    curr_original_frame = 0
    for i in range(required_frame_length):
        if i in idx_to_interpolate:
            fin_array[i] = (fin_array[i - 1] + video_embedding[curr_original_frame]) / 2
        else:
            fin_array[i] = video_embedding[curr_original_frame]
            curr_original_frame += 1
    assert (
        curr_original_frame == video_embedding.shape[0]
    ), "Didn't iterate through all original frames"

    return fin_array


def reduce_frames(video_embedding, required_frame_length):
    if required_frame_length == 1:
        return np.vstack([video_embedding[0]])

    diff_between_frames = video_embedding[1:] - video_embedding[:-1]
    max_movement = np.max(diff_between_frames, axis=1)
    max_movement_indexes = np.argsort(max_movement)[::-1] + 1

    required_max_movement_frames = video_embedding[
        sorted(max_movement_indexes[: required_frame_length - 1])
    ]

    return np.vstack([video_embedding[0], required_max_movement_frames])


def convert_to_required_number_of_frames(video_embedding, required_frame_length):
    num_frames = video_embedding.shape[0]
    if num_frames == required_frame_length:
        return video_embedding
    elif num_frames > required_frame_length:
        return reduce_frames(video_embedding, required_frame_length)
    else:
        return interpolate_frames(video_embedding, required_frame_length)


def operate_on_file(source_file, frame_length):
    data = np.load(source_file)
    return convert_to_required_number_of_frames(data, frame_length)


def preprocess_dir(source_dir, destination_dir, frame_length):
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
                keypoints = operate_on_file(source_path, frame_length)
                np.save(destination_path, keypoints)


SOURCE_DIR = "/home/sn0w/Desktop/IIITA/Sem7/mini/data/custom/signs_extracted_keypoint"
DESTINATION_DIR = (
    "/home/sn0w/Desktop/IIITA/Sem7/mini/data/custom/signs_same_length_keypoint"
)
NUM_FRAMES = 60
preprocess_dir(
    source_dir=SOURCE_DIR, destination_dir=DESTINATION_DIR, frame_length=NUM_FRAMES
)
