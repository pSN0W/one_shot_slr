import mlflow
from itertools import product

mlflow.set_tracking_uri("http://localhost:5000")

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

EPOCHS = 30
FEATURE_DIM = 258
MAX_SEQUENCE_LENGTH = 60
DROPOUTS = 0.2

ENCODING_STEPS = [512] * 4
FCC = [2048] * 2
LEARNING_RATE = 1e-6
TRAIN_BATCH_SIZES = 16

DROPOUTS = 0.1
MARGIN = 0.25

NUM_HEADS = 4
EXPANSION = 2
EVAL_BATCH_SIZE = 64
TRAIN_DATA_LOC = (
    "/home/sn0w/Desktop/IIITA/Sem7/mini/data/include_same_length_keypoint_augmented"
)
EVAL_DATA_LOC = (
    "/home/sn0w/Desktop/IIITA/Sem7/mini/data/include_same_length_keypoint_test"
)


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


def run_epoch(loader, model, optimizer, device, margin):
    epoch_loss = []

    pbar = tqdm(loader)
    for set1, set2 in pbar:
        # zero the optimizer
        optimizer.zero_grad()

        # move the input to device
        set1, set2 = set1.to(device), set2.to(device)
        set1_encoded, set2_encoded = model(set1, set2)

        # compute loss
        loss = TripletLossFn(set1_encoded, set2_encoded, device, margin)

        # optimize the parameters
        loss.backward()
        optimizer.step()

        epoch_loss.append(loss.cpu().item())

        pbar.set_description(f"Training {round(epoch_loss[-1],4)} >> ")

    return sum(epoch_loss) / len(epoch_loss)


def run_experiment(
    train_batch_size,
    eval_batch_size,
    encoding_steps,
    fcc,
    num_head,
    dropout,
    learning_rate,
    margin,
    **kwargs,
):
    device = torch.device("cuda:0")
    # device = torch.device("cpu")

    train_evaluate_dataset = SiameseValidationDataset(data_loc=TRAIN_DATA_LOC)
    train_eval_loader = DataLoader(
        dataset=train_evaluate_dataset, batch_size=eval_batch_size, shuffle=False
    )

    test_evaluate_dataset = SiameseValidationDataset(data_loc=EVAL_DATA_LOC)
    test_eval_loader = DataLoader(
        dataset=test_evaluate_dataset, batch_size=eval_batch_size, shuffle=False
    )

    dataloader = SiameseLandmarkDataloader(
        data_loc=TRAIN_DATA_LOC, batch_size=train_batch_size
    )

    model = KeypointSiameseNetwork(
        feature_dim=FEATURE_DIM,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
        encoders=encoding_steps,
        fcc=fcc,
        num_heads=num_head,
        dropout_p=dropout,
        expansion=EXPANSION,
    ).to(device)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    best_test_acc = 0

    for i in range(EPOCHS):
        model.train()
        loss = run_epoch(
            loader=dataloader,
            model=model,
            optimizer=optimizer,
            device=device,
            margin=margin,
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
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            mlflow.pytorch.log_model(model, artifact_path=f"{i}_{best_test_acc}")

        mlflow.log_metric("train_loss", round(loss, 4), step=i + 1)
        mlflow.log_metric("train_acc", train_acc, step=i + 1)
        mlflow.log_metric("test_acc", test_acc, step=i + 1)
        print(
            f"Epoch {i+1} Loss {round(loss,4)} Train Accuracy {train_acc} Test Accuracy {test_acc}"
        )
    return best_test_acc


def create_experiment_and_run(params, experiment_name):
    experiment_id = mlflow.get_experiment_by_name(experiment_name)
    if experiment_id is None:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment_id.experiment_id
    run_name = f"{params['train_batch_size']}_epoch_augmented"
    with mlflow.start_run(
        experiment_id=experiment_id, run_name=run_name
    ) as mlflow_client:
        mlflow.log_params(params)
        run_experiment(**params)


if __name__ == "__main__":
    params = dict(
        eval_batch_size=EVAL_BATCH_SIZE,
        train_batch_size=TRAIN_BATCH_SIZES,
        learning_rate=LEARNING_RATE,
        dropout=DROPOUTS,
        num_head=NUM_HEADS,
        encoding_steps=ENCODING_STEPS,
        fcc=FCC,
        margin=MARGIN,
    )
    create_experiment_and_run(params=params, experiment_name="models")
