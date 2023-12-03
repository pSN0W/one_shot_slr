import cv2
import mediapipe as mp
import numpy as np
import os


mp_holistic = mp.solutions.holistic

SOURCE_DIR = "/home/sn0w/Desktop/IIITA/Sem7/mini/data/custom/signs"
DESTINATION_DIR = (
    "/home/sn0w/Desktop/IIITA/Sem7/mini/data/custom/signs_extracted_keypoint"
)


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False  # Image is no longer writeable
    results = model.process(image)  # Make prediction
    image.flags.writeable = True  # Image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # COLOR COVERSION RGB 2 BGR
    return image, results


def extract_keypoints(results):
    pose = (
        np.array(
            [
                [res.x, res.y, res.z, res.visibility]
                for res in results.pose_landmarks.landmark
            ]
        ).flatten()
        if results.pose_landmarks
        else np.zeros(33 * 4)
    )
    lh = (
        np.array(
            [[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]
        ).flatten()
        if results.left_hand_landmarks
        else np.zeros(21 * 3)
    )
    rh = (
        np.array(
            [[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]
        ).flatten()
        if results.right_hand_landmarks
        else np.zeros(21 * 3)
    )
    return np.concatenate([pose, lh, rh])


def extract_keypoint_from_mov(video_path):
    cap = cv2.VideoCapture(video_path)
    vid_landmarks = []
    # Set mediapipe model
    with mp_holistic.Holistic(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as holistic:
        while True:
            # Read feed
            ret, frame = cap.read()

            # Check if the frame was read successfully
            if not ret:
                break

            # Make detections
            image, results = mediapipe_detection(frame, holistic)
            keypoint = extract_keypoints(results)
            vid_landmarks.append(keypoint)
    return np.array(vid_landmarks)


def extract_keypoint_for_dir(source_dir, destination_dir):
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
                keypoints = extract_keypoint_from_mov(source_path)
                np.save(destination_path, keypoints)


extract_keypoint_for_dir(source_dir=SOURCE_DIR, destination_dir=DESTINATION_DIR)
