import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import itertools
import pandas as pd
import numpy as np
import pandas as pd
import random

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

import ssl
import sys
import os

ssl._create_default_https_context = ssl._create_unverified_context

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from data_handling.data_loading import league_data_loader


# Data Preparation Class
class SequenceDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence, ext_data, fga_weight, label = self.sequences[idx]
        return (
            torch.FloatTensor(sequence),
            torch.FloatTensor(ext_data),
            torch.FloatTensor([fga_weight]),
            torch.FloatTensor([label]),
        )


def create_inout_sequences(input_data, features, response, start_index, lag=4):
    inout_seq = []
    input_data = input_data.reset_index(drop=True)
    for i in range(start_index, len(input_data)):
        end = i + lag
        if end > len(input_data) - 1:
            break
        seq_data = input_data.loc[i : (end - 1), response].values.astype(float)
        ext_data = input_data.loc[end, features].values.astype(float)
        fga_weight = input_data.loc[end, "weights"]
        label = input_data.loc[end, response]
        inout_seq.append((seq_data, ext_data, fga_weight, label))
    return inout_seq


def create_weights(input_data, weight_feature, seasons):
    input_data["weights"] = (
        input_data[weight_feature].values
        / input_data.loc[input_data["season"].isin(seasons), weight_feature].sum()
    )
    return input_data


def prepare_data(
    data, train_seasons, val_seasons, test_seasons, features, response, weight_feature
):
    full_data = data.copy()

    # Assign start indices for val and test data
    # start train index at first game of third season
    train_start_index = full_data[
        full_data["season"] == (min(train_seasons) + 2)
    ].index.min()
    val_start_index = full_data[full_data["season"].isin(val_seasons)].index.min()
    test_start_index = full_data[full_data["season"].isin(test_seasons)].index.min()

    train_data = full_data[full_data["season"].isin(train_seasons)]
    val_data = full_data[full_data["season"].isin(train_seasons + val_seasons)]
    test_data = full_data[
        full_data["season"].isin(train_seasons + val_seasons + test_seasons)
    ]

    # scale features - use train data min and max
    feature_min = train_data[features].min()
    feature_max = train_data[features].max()
    train_data[features] = (train_data[features] - feature_min) / (
        feature_max - feature_min
    )
    val_data[features] = (val_data[features] - feature_min) / (
        feature_max - feature_min
    )
    test_data[features] = (test_data[features] - feature_min) / (
        feature_max - feature_min
    )

    # Create weights
    # make sure min in train_seasons is min(train_seasons) + 2
    min_season = min(train_seasons) + 2
    train_data = create_weights(
        train_data, weight_feature, list(range(min_season, max(train_seasons) + 1))
    )
    val_data = create_weights(val_data, weight_feature, val_seasons)
    test_data = create_weights(test_data, weight_feature, test_seasons)

    # Sequences
    train_seq = create_inout_sequences(train_data, features, response, start_index=1)
    val_seq = create_inout_sequences(
        val_data, features, response, start_index=val_start_index
    )
    test_seq = create_inout_sequences(
        test_data, features, response, start_index=test_start_index
    )

    # Create datasets
    train_dataset = SequenceDataset(train_seq)
    val_dataset = SequenceDataset(val_seq)
    test_dataset = SequenceDataset(test_seq)

    return train_dataset, val_dataset, test_dataset


# Data Loaders Function
def create_data_loaders(train_dataset, val_dataset, test_dataset, batch_size):
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    return train_loader, val_loader, test_loader


def collate_fn(batch):
    sequences, ext_data, weights, labels = zip(*batch)
    lengths = [len(seq) for seq in sequences]
    sequences_padded = pad_sequence(sequences, batch_first=True)
    ext_data = torch.stack(ext_data)
    weights = torch.stack(weights)
    labels = torch.stack(labels)
    return sequences_padded, ext_data, weights, labels, lengths


# Custom Weighted MSE Loss
def weighted_mse_loss(input, target, weight):
    return (weight * (input - target) ** 2).sum() / weight.sum()


# LSTM Model Class with Pack Padded Sequence
class LSTMModel(nn.Module):
    def __init__(self, feature_size, hidden_size, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            4, hidden_size, num_layers, batch_first=True, bidirectional=False
        )
        self.linear = nn.Linear(1 * num_layers * hidden_size + feature_size, 4)
        self.act = nn.ReLU()
        self.final_linear = nn.Linear(4, 1)

    def forward(self, x, ext_data, lengths):
        packed = pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        packed_output, (ht, ct) = self.lstm(x)

        # Get the last output of the LSTM
        last_outputs = ht.view(1, -1)

        # scaled_lstm_outputs = self.relu(self.linear_lstm(last_outputs))
        # scaled_ext_data = self.relu(self.linear_ext(ext_data))

        combined = torch.cat((last_outputs, ext_data), dim=1)
        prediction = self.final_linear(self.act(self.linear(combined)))
        return prediction.squeeze()


# Training Function
def train_model(model, train_loader, val_loader, optimizer, epochs):
    best_val_loss = float("inf")
    best_model = None
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for seq, ext_data, weights, labels, lengths in train_loader:
            optimizer.zero_grad()
            y_pred = model(seq, ext_data, lengths)
            loss = weighted_mse_loss(y_pred, labels, weights)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        val_loss = evaluate_model(model, val_loader)
        print(
            f"Epoch {epoch+1}/{epochs}, Training Loss: {total_train_loss}, Validation Loss: {val_loss}"
        )

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model
    return best_val_loss, best_model


# Evaluation Function
def evaluate_model(model, loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for seq, ext_data, weights, labels, lengths in loader:
            y_pred = model(seq, ext_data, lengths)
            loss = weighted_mse_loss(y_pred, labels, weights)
            total_loss += loss.item()
    return total_loss


def hyperparameter_tuning(
    data, train_seasons, val_seasons, test_seasons, features, response, param_grid=None
):
    # Default hyperparameters
    default_params = {
        "hidden_size": 64,
        "num_layers": 2,
        "lr": 1e-3,
        "batch_size": 1,
        "epochs": 500,
    }

    if param_grid is None or not param_grid:
        # Use default hyperparameters
        param_grid = {k: [v] for k, v in default_params.items()}

    best_model = None
    best_params = {}
    best_val_loss = float("inf")

    for params in itertools.product(*param_grid.values()):
        param_dict = dict(zip(param_grid.keys(), params))
        print("Testing with parameters:", param_dict)
        model = LSTMModel(
            feature_size=len(features),
            hidden_size=param_dict["hidden_size"],
            num_layers=param_dict["num_layers"],
        )
        optimizer = optim.Adam(model.parameters(), lr=param_dict["lr"])

        train_dataset, val_dataset, test_dataset = prepare_data(
            data,
            train_seasons,
            val_seasons,
            test_seasons,
            features,
            response,
            weight_feature="fga",
        )
        train_loader, val_loader, test_dataset = create_data_loaders(
            train_dataset,
            val_dataset,
            test_dataset,
            batch_size=param_dict["batch_size"],
        )

        val_loss, model = train_model(
            model, train_loader, val_loader, optimizer, param_dict["epochs"]
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_params = param_dict
            best_model = model

    print(f"Best Parameters: {best_params}, Best Validation Loss: {best_val_loss}")
    return best_model, best_params


# Function to Run the Model with Best Parameters and Save Test Predictions
def run_and_save_predictions(
    data, train_seasons, val_seasons, test_seasons, features, response, best_params
):
    train_dataset, val_dataset, test_dataset = prepare_data(
        data,
        train_seasons,
        val_seasons,
        test_seasons,
        features,
        response,
        weight_feature="fga",
    )
    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset, val_dataset, test_dataset, batch_size=best_params["batch_size"]
    )

    model = LSTMModel(
        feature_size=len(features),
        hidden_size=best_params["hidden_size"],
        num_layers=best_params["num_layers"],
    )
    model.load_state_dict(torch.load("best_model.pth"))

    # Save Test Predictions
    model.eval()
    test_predictions = []
    with torch.no_grad():
        for seq, ext_data, weights, labels, lengths in test_loader:
            y_pred = model(seq, ext_data, lengths)
            test_predictions += [y_pred.tolist()]
    out = pd.concat(
        [
            data[data.season.isin(test_seasons)][
                ["league_avg_fg3a_fga", "fga"]
            ].reset_index(drop=True),
            pd.DataFrame(test_predictions, columns=["Predictions"]).reset_index(drop=True)
        ],
        axis=1
    )
    # Account for drift from train to val to test
    out["Predictions"] =  out.Predictions.shift(4) - out.Predictions.mean() + 2 * data[data.season.isin(val_seasons)].league_avg_fg3a_fga.mean() - data[data.season.isin(train_seasons)].league_avg_fg3a_fga.mean()
    out.to_csv("lstm_test_predictions.csv", index=False)
    print("Test predictions saved to 'lstm_test_predictions.csv'.")


# Run the Model with Best Parameters and Save Test Predictions
data = league_data_loader(seasons=list(range(2010, 2020)))
features = ["season_type", "date_num"]
best_model, best_params = hyperparameter_tuning(
    data,
    list(range(2010, 2015)),
    list(range(2015, 2016)),
    list(range(2016, 2020)),
    features,
    "league_avg_fg3a_fga",
    param_grid=None,
)
torch.save(best_model.state_dict(), "best_model.pth")
run_and_save_predictions(
    data,
    list(range(2010, 2015)),
    list(range(2015, 2016)),
    list(range(2016, 2020)),
    features,
    "league_avg_fg3a_fga",
    best_params,
)

df_lstm = pd.read_csv("lstm_test_predictions.csv")


def weighted_mse(true, pred, weights):
    return (weights * (true - pred) ** 2).sum() / weights.sum()


print(
    weighted_mse(
        df_lstm.league_avg_fg3a_fga.iloc[4:].reset_index(drop=True),
        df_lstm.Predictions.iloc[4:].reset_index(drop=True),
        df_lstm.fga.iloc[4:].reset_index(drop=True),
    )
)
