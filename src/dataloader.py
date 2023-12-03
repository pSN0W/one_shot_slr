import os
from pathlib import Path
import numpy as np
import random
import torch

from torch.utils.data import Dataset


def generate_file_dic_for_data(data_loc):
    file_dic = {}

    for root, dirs, files in os.walk(data_loc):
        # Copy files from the source directory to the destination directory
        for file_name in files:
            source_path = Path(os.path.join(root, file_name))
            file_dic.setdefault(int(source_path.parent.stem), [])
            file_dic[int(source_path.parent.stem)].append(source_path)

    return file_dic


def get_relative_to_nose(arr):
    pose = arr[:, : 33 * 4].reshape(-1, 33, 4)

    other = arr[:, 33 * 4 :].reshape(60, -1, 3)
    nose = pose[0, 0].copy().reshape(1, -1)
    nose[0, -1] = 0
    transformed_pose = pose - nose
    transformed_other = other - nose[:, :-1]
    return np.hstack(
        [transformed_pose.reshape(60, -1), transformed_other.reshape(60, -1)]
    )


def load_keypoint(keypoint_path):
    data = get_relative_to_nose(np.load(keypoint_path))
    return torch.tensor(data, dtype=torch.float32)


class SiameseValidationDataset(Dataset):
    def __init__(self, data_loc) -> None:
        file_dic = generate_file_dic_for_data(data_loc)
        file_with_label = {v: k for k, lst in file_dic.items() for v in lst}
        self.file_loc = list(file_with_label.keys())
        self.labels = list(file_with_label.values())

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = load_keypoint(self.file_loc[idx])
        y = self.labels[idx]
        return x, y


class SiameseFeatureExtractorDataset(SiameseValidationDataset):
    def __init__(self, data_loc) -> None:
        file_dic = generate_file_dic_for_data(data_loc)
        one_sample_file_dic = {k: v[0] for k, v in file_dic.items()}
        self.file_loc = list(one_sample_file_dic.values())
        self.labels = list(one_sample_file_dic.keys())


class SiameseLandmarkDataloader:
    def __init__(self, data_loc, batch_size, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.current_idx = -1

        self.file_dic = generate_file_dic_for_data(data_loc)
        self.files = [
            file_name
            for file_in_dic in self.file_dic.values()
            for file_name in file_in_dic
        ]
        if self.shuffle:
            random.shuffle(self.files)

    def __len__(self):
        return len(self.files)

    def __iter__(self):
        return self

    def __next__(self):
        self.current_idx += 1

        if self.current_idx == len(self.files):
            self.current_idx = -1
            if self.shuffle:
                random.shuffle(self.files)
            raise StopIteration

        return self.get_batch_for_anchor(self.files[self.current_idx], self.batch_size)

    def get_batch_for_anchor(self, anchor_path, batch_size):
        landmark_paths1, landmark_paths2 = self.get_batch_paths_for_anchor(
            anchor_path, batch_size
        )
        batch_set1 = torch.stack([load_keypoint(path) for path in landmark_paths1])
        batch_set2 = torch.stack([load_keypoint(path) for path in landmark_paths2])
        return batch_set1, batch_set2

    def get_positive_example(self, anchor_path):
        class_label = int(anchor_path.parent.stem)
        possible_positive_example = random.choice(self.file_dic[class_label])
        while anchor_path == possible_positive_example:
            possible_positive_example = random.choice(self.file_dic[class_label])
        return possible_positive_example

    def get_random_negative_example(self, anchor_path, required_examples):
        class_label = int(anchor_path.parent.stem)
        possible_classes = set(self.file_dic.keys()) - set([class_label])

        assert required_examples <= len(
            possible_classes
        ), "Batch size can't be more then number of available classes"

        landmark_set1 = []
        landmark_set2 = []

        for classes in random.sample(list(possible_classes), required_examples):
            example1, example2 = random.sample(self.file_dic[classes], 2)
            landmark_set1.append(example1)
            landmark_set2.append(example2)

        return landmark_set1, landmark_set2

    def get_batch_paths_for_anchor(self, anchor_path, batch_size):
        (
            negative_landmark_set1,
            negative_landmark_set2,
        ) = self.get_random_negative_example(anchor_path, batch_size - 1)
        positive_example = self.get_positive_example(anchor_path)
        return [anchor_path] + negative_landmark_set1, [
            positive_example
        ] + negative_landmark_set2
