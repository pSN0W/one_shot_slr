import numpy as np
import einops
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def get_prediction(feature_space, current_embed):
    labels = list(feature_space.keys())
    feature_encoding = np.vstack(list(feature_space.values()))
    similarity = cosine_distance(feature_encoding, current_embed)
    predictions = np.argmax(similarity, axis=-1)
    return [labels[p] for p in predictions]


def cosine_distance(feature_encoding: np.ndarray, curr_feature: np.ndarray):
    return einops.einsum(
        feature_encoding, curr_feature, "num_possible d, b d -> b num_possible"
    )


class MetricsComputer:
    def __init__(self, feature_encoding) -> None:
        self.feature_encoding = feature_encoding
        self.labels = list(self.feature_encoding.keys())
        self.confusion_matrix = np.zeros(
            (len(self.labels), len(self.labels)), dtype=int
        )
        self.label_to_idx = {l: i for i, l in enumerate(self.labels)}

    def update(self, actual, embedding):
        preds = get_prediction(self.feature_encoding, embedding)
        for l, p in zip(actual, preds):
            self.confusion_matrix[self.label_to_idx[l]][self.label_to_idx[p]] += 1

    def reset(self):
        self.confusion_matrix = np.zeros(
            (len(self.labels), len(self.labels)), dtype=int
        )

    def compute_accuracy(self):
        correct = np.diag(self.confusion_matrix).sum()
        total = self.confusion_matrix.sum()
        return round(correct * 100 / total, 4)

    def plot(self, loc):
        df = pd.DataFrame(
            data=self.confusion_matrix, columns=self.labels, index=self.labels
        )
        fig = plt.figure(figsize=(16,16))
        sns.heatmap(df, annot=True, cmap="viridis")


        plt.title("Confusion Matrix")
        plt.xlabel("Predicted Class")
        plt.ylabel("True Class")
        fig.savefig(loc)
