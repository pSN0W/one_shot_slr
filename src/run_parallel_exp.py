import mlflow

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

EPOCHS = 10
FEATURE_DIM = 258
MAX_SEQUENCE_LENGTH = 60
DROPOUTS = 0.2

ENCODING_WIDTHS = [128, 256, 512]
ENCODIING_DEPTH = [2, 4, 6]
FCC_WIDTH = [512, 1024, 2048]
FCC_DEPTH = [2, 4, 6]
LEARNING_RATE = 0.00001
MARGIN = 0.25
TRAIN_BATCH_SIZES = 16

NUM_HEADS = 4
EXPANSION = 2
EVAL_BATCH_SIZE = 64
TRAIN_DATA_LOC = "/home/sn0w/Desktop/IIITA/Sem7/mini/data/include_same_length_keypoint"
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
    **kwargs
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
        loss = run_epoch(
            loader=dataloader, model=model, optimizer=optimizer, device=device, margin=margin
        )

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
        best_test_acc = max([test_acc, best_test_acc])

        mlflow.log_metric("train_loss", round(loss, 4),step=i+1)
        mlflow.log_metric("train_acc", train_acc,step=i+1)
        mlflow.log_metric("test_acc", test_acc,step=i+1)
    return best_test_acc


def create_experiment_and_run(params, to_update, update_values):
    experiment_name = f"{to_update}_effect"
    experiment_id = mlflow.get_experiment_by_name(experiment_name)
    if experiment_id is None:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment_id.experiment_id
    best_value = -1
    best_accuracy = 0
    for val in update_values:
        run_name = f"{to_update}_{val}"
        params[to_update] = val
        with mlflow.start_run(
            experiment_id=experiment_id, run_name=run_name
        ) as mlflow_client:
            mlflow.log_params(params)
            acc = run_experiment(
                mlflow_client=mlflow_client, **transform_params(params)
            )
            if acc > best_accuracy:
                best_accuracy = acc
                best_value = val

    return best_value


def transform_params(params: dict):
    encoding_width = params.get('encoding_width')
    encoding_depth = params.get('encoding_depth')
    fcc_width = params.get('fcc_width')
    fcc_depth = params.get('fcc_depth')
    params["encoding_steps"] = [encoding_width] * encoding_depth
    params["fcc"] = [fcc_width] * fcc_depth
    return params



if __name__ == "__main__":
    best_ew = ENCODING_WIDTHS[0]
    best_ed = ENCODIING_DEPTH[0]
    best_fw = FCC_WIDTH[0]
    best_fd = FCC_DEPTH[0]

    params = dict(
        train_batch_size=16,
        eval_batch_size=64,
        encoding_width=ENCODING_WIDTHS[0],
        encoding_depth=ENCODIING_DEPTH[0],
        fcc_width=FCC_WIDTH[0],
        fcc_depth=FCC_DEPTH[0],
        num_head=NUM_HEADS,
        dropout=0.2,
        learning_rate=LEARNING_RATE,
        margin=MARGIN
    )
    params["encoding_width"] = create_experiment_and_run(params, "encoding_width", ENCODING_WIDTHS)
    params["encoding_depth"] = create_experiment_and_run(params, "encoding_depth", ENCODIING_DEPTH)
    params["fcc_width"] = create_experiment_and_run(params, "fcc_width", FCC_WIDTH)
    params["fcc_depth"] = create_experiment_and_run(params, "fcc_depth", FCC_DEPTH)
