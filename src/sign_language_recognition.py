import mlflow

mlflow.set_tracking_uri("http://localhost:5000")

import torch

from dataloader import load_keypoint
from test_utils import get_prediction


class SignLanguageModel:
    def __init__(self, model_path) -> None:
        self.model = mlflow.pytorch.load_model(model_path)
        self.model.eval()
        self.device = torch.device("cuda:0")
        self.encodings = {}

    def add_new_sign(self, sign_name, file_path):
        self.encodings[sign_name] = self.encode(file_path)

    def predict(self, file_path):
        encoded = self.encode(file_path).reshape(1, -1)
        return get_prediction(self.encodings, encoded)

    def encode(self, vid_file_path):
        file_loc = self.get_file_path(vid_file_path)
        with torch.no_grad():
            keypoint = load_keypoint(file_loc)
            add_batch_dim = keypoint.unsqueeze(0)
            batch = add_batch_dim.to(self.device)
            return self.model(batch)[0].cpu().numpy()

    def get_file_path(self, pth):
        pth = str(pth)
        pth = pth.replace("signs", "signs_same_length_keypoint").replace(".mp4", ".npy")
        return pth
