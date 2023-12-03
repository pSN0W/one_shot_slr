import mlflow
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader import (
    SiameseLandmarkDataloader,
    SiameseFeatureExtractorDataset,
    SiameseValidationDataset,
)
from loss import TripletLossFn
from model import KeypointSiameseNetwork
from test_utils import MetricsComputer

DROPOUT = 0.3
EPOCHS = 50
FEATURE_DIM = 258
MAX_SEQUENCE_LENGTH = 60
ENCODING_STEPS = [512]*4
FCC = [2048]*2
NUM_HEADS = 4
EXPANSION = 2
LEARNING_RATE = 0.00001
MARGIN = 0.3
TRAIN_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 64
TRAIN_DATA_LOC = "/home/sn0w/Desktop/IIITA/Sem7/mini/data/include_same_length_keypoint"
EVAL_DATA_LOC = "/home/sn0w/Desktop/IIITA/Sem7/mini/data/include_same_length_keypoint_test"


def create_feature_encoding(data_loc, model, device):
    dataset = SiameseFeatureExtractorDataset(data_loc)
    dataloader = DataLoader(dataset=dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False)
    feature_embedding = {}
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.numpy()
            embedding = model(x).cpu().numpy()
            for label, embed in zip(y, embedding):
                feature_embedding[label] = embed
    return feature_embedding


def run_epoch(loader, model, optimizer, device):
    epoch_loss = []

    pbar = tqdm(loader)
    for set1, set2 in pbar:
        # zero the optimizer
        optimizer.zero_grad()

        # move the input to device
        set1, set2 = set1.to(device), set2.to(device)
        set1_encoded, set2_encoded = model(set1, set2)

        # compute loss
        loss = TripletLossFn(set1_encoded, set2_encoded, device, MARGIN)

        # optimize the parameters
        loss.backward()
        optimizer.step()

        epoch_loss.append(loss.cpu().item())

        pbar.set_description(f"Training {round(epoch_loss[-1],4)} >> ")

    return sum(epoch_loss) / len(epoch_loss)


if __name__ == "__main__":
    device = torch.device("cuda:0")
    #device = torch.device("cpu")

    train_evaluate_dataset = SiameseValidationDataset(data_loc=TRAIN_DATA_LOC)
    train_eval_loader = DataLoader(
        dataset=train_evaluate_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False
    )

    test_evaluate_dataset = SiameseValidationDataset(data_loc=EVAL_DATA_LOC)
    test_eval_loader = DataLoader(
        dataset=test_evaluate_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False
    )

    dataloader = SiameseLandmarkDataloader(
        data_loc=TRAIN_DATA_LOC, batch_size=TRAIN_BATCH_SIZE
    )

    model = KeypointSiameseNetwork(
        feature_dim=FEATURE_DIM,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
        encoders=ENCODING_STEPS,
        fcc=FCC,
        num_heads=NUM_HEADS,
        dropout_p=DROPOUT,
        expansion=EXPANSION,
    ).to(device)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

    for i in range(EPOCHS):
        model.train()
        loss = run_epoch(
            loader=dataloader, model=model, optimizer=optimizer, device=device
        )
        
        model.eval()

        feature_embedding = create_feature_encoding(
            data_loc=TRAIN_DATA_LOC, model=model, device=device
        )
        train_metrics_computer = MetricsComputer(feature_encoding=feature_embedding)
        with torch.no_grad():
            for x, y in train_eval_loader:
                x, y = x.to(device), y.numpy()
                embed = model(x).cpu().numpy()
                train_metrics_computer.update(y, embed)
        train_acc = train_metrics_computer.compute_accuracy()

        # Validation
        feature_embedding.update(
            create_feature_encoding(data_loc=EVAL_DATA_LOC, model=model, device=device)
        )
        test_metrics_computer = MetricsComputer(feature_encoding=feature_embedding)
        with torch.no_grad():
            for x, y in test_eval_loader:
                x, y = x.to(device), y.numpy()
                embed = model(x).cpu().numpy()
                test_metrics_computer.update(y, embed)
        test_acc = test_metrics_computer.compute_accuracy()

        print(
            f"Epoch {i+1} Loss {round(loss,4)} Train Accuracy {train_acc} Test Accuracy {test_acc}"
        )
    train_metrics_computer.plot("train_confusion.png")
    test_metrics_computer.plot("test_confusion.png")
