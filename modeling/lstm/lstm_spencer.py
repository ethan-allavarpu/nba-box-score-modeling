import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from modeling.data_handling.data_loading import league_data_loader
import itertools
import pandas as pd
from sklearn.metrics import r2_score

# Data Preparation Class
class SequenceDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence, ext_data, fga_weight, label = self.sequences[idx]
        return torch.FloatTensor(sequence), torch.FloatTensor(ext_data), torch.FloatTensor([fga_weight]), torch.FloatTensor([label])

def create_inout_sequences(input_data, features, response, start_index, weight_feature, response_mean, response_std):
    inout_seq = []
    input_data = input_data.reset_index(drop=True)
    weight_sum = 0
    for i in range(max(start_index, 10), len(input_data)):
        seq_data = input_data.loc[i-10:i-1, features + [response]].values
        ext_data = input_data.loc[i, features].values.astype(float)
        fga_weight = input_data.loc[i, weight_feature]
        label = input_data.loc[i, response] * response_std + response_mean
        inout_seq.append((seq_data, ext_data, fga_weight, label))
        weight_sum += fga_weight
    return inout_seq, weight_sum


def prepare_data(data, train_seasons, val_seasons, test_seasons, features, response, weight_feature):
    full_data = data.copy()

    # Assign start indices for val and test data
    # start train index at first game of fourth season
    train_start_index = full_data[full_data['season'] == (min(train_seasons) + 3)].index.min()
    val_start_index = full_data[full_data['season'].isin(val_seasons)].index.min()
    test_start_index = full_data[full_data['season'].isin(test_seasons)].index.min()

    train_data = full_data[full_data['season'].isin(train_seasons)]
    val_data = full_data[full_data['season'].isin(train_seasons + val_seasons)]
    test_data = full_data[full_data['season'].isin(train_seasons + val_seasons + test_seasons)]

    # scale features - use train data mean and std
    feature_mean = train_data[features].mean()
    feature_std = train_data[features].std()
    train_data[features] = (train_data[features] - feature_mean) / feature_std
    val_data[features] = (val_data[features] - feature_mean) / feature_std
    test_data[features] = (test_data[features] - feature_mean) / feature_std

    # scale response - use train data mean and std
    response_mean = train_data[response].mean()
    response_std = train_data[response].std()
    train_data[response] = (train_data[response] - response_mean) / response_std
    val_data[response] = (val_data[response] - response_mean) / response_std
    test_data[response] = (test_data[response] - response_mean) / response_std

    # Sequences
    train_seq, train_weight_sum = create_inout_sequences(train_data, features, response, start_index=1, weight_feature=weight_feature, response_mean=response_mean, response_std=response_std)
    val_seq, val_weight_sum = create_inout_sequences(val_data, features, response, start_index=val_start_index, weight_feature=weight_feature,  response_mean=response_mean, response_std=response_std)
    test_seq, test_weight_sum = create_inout_sequences(test_data, features, response, start_index=test_start_index,  weight_feature=weight_feature,  response_mean=response_mean, response_std=response_std)

    # Normalize weights
    train_seq = [(seq, ext_data, weight / train_weight_sum, label) for seq, ext_data, weight, label in train_seq]
    val_seq = [(seq, ext_data, weight / val_weight_sum, label) for seq, ext_data, weight, label in val_seq]
    test_seq = [(seq, ext_data, weight / test_weight_sum, label) for seq, ext_data, weight, label in test_seq]

    # Create datasets
    train_dataset = SequenceDataset(train_seq)
    val_dataset = SequenceDataset(val_seq)
    test_dataset = SequenceDataset(test_seq)

    return train_dataset, val_dataset, test_dataset

# Data Loaders Function
def create_data_loaders(train_dataset, val_dataset, test_dataset, batch_size):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
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
    return (weight * (input - target) ** 2).sum()
    # return ((input - target) ** 2).mean()

# LSTM Model Class with Pack Padded Sequence
class LSTMModel(nn.Module):
    def __init__(self, feature_size, hidden_size, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(feature_size + 1, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.final_linear = nn.Linear(2 * num_layers * hidden_size + feature_size, 1)

    def forward(self, x, ext_data, lengths):
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_output, (ht, ct) = self.lstm(x)

        # Get the last output of the LSTM
        last_outputs = ht.view(1, -1)

        # scaled_lstm_outputs = self.relu(self.linear_lstm(last_outputs))
        # scaled_ext_data = self.relu(self.linear_ext(ext_data))

        combined = torch.cat((last_outputs, ext_data), dim=1)
        prediction = self.final_linear(combined)
        return prediction

# Training Function
def train_model(model, train_loader, val_loader, optimizer, epochs):
    best_val_loss = float('inf')
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
        print(f'Epoch {epoch+1}/{epochs}, Training Loss: {total_train_loss}, Validation Loss: {val_loss}')

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

def hyperparameter_tuning(data, train_seasons, val_seasons, test_seasons, features, response, param_grid=None):
    # Default hyperparameters
    default_params = {
        'hidden_size': 64,
        'num_layers': 2,
        'lr': 5e-4,
        'batch_size': 1,
        'epochs': 200
    }

    if param_grid is None or not param_grid:
        # Use default hyperparameters
        param_grid = {k: [v] for k, v in default_params.items()}

    best_model = None
    best_params = {}
    best_val_loss = float('inf')
    
    for params in itertools.product(*param_grid.values()):
        param_dict = dict(zip(param_grid.keys(), params))
        print("Testing with parameters:", param_dict)
        model = LSTMModel(feature_size=len(features), hidden_size=param_dict['hidden_size'], num_layers=param_dict['num_layers'])
        optimizer = optim.Adam(model.parameters(), lr=param_dict['lr'])

        train_dataset, val_dataset, test_dataset = prepare_data(data, train_seasons, val_seasons, test_seasons, features, response, weight_feature='fga')
        train_loader, val_loader, test_dataset = create_data_loaders(train_dataset, val_dataset, test_dataset, batch_size=param_dict['batch_size'])

        val_loss, model = train_model(model, train_loader, val_loader, optimizer, param_dict['epochs'])

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_params = param_dict
            best_model = model

    print(f"Best Parameters: {best_params}, Best Validation Loss: {best_val_loss}")
    return best_model, best_params

# Function to Run the Model with Best Parameters and Save Test Predictions
def run_and_save_predictions(data, train_seasons, val_seasons, test_seasons, features, response, best_params):
    train_dataset, val_dataset, test_dataset = prepare_data(data, train_seasons, val_seasons, test_seasons, features, response, weight_feature='fga')
    train_loader, val_loader, test_loader = create_data_loaders(train_dataset, val_dataset, test_dataset, batch_size=best_params['batch_size'])

    model = LSTMModel(feature_size=len(features), hidden_size=best_params['hidden_size'], num_layers=best_params['num_layers'])
    model.load_state_dict(torch.load('best_model.pth'))

    # Save Test Predictions
    model.eval()
    test_predictions = []
    with torch.no_grad():
        for seq, ext_data, weights, labels, lengths in test_loader:
            y_pred = model(seq, ext_data, lengths)
            test_predictions.extend(y_pred.tolist())

    pred_data = data[data['season'].isin(test_seasons)][['game_date', response]]
    pred_data[f'predicted_{response}'] = [p[0] for p in test_predictions]
    pred_data.to_csv('test_predictions.csv', index=False)
    print("Test predictions saved to 'test_predictions.csv'.")
    # print test R squared
    print("Test R squared:", r2_score(pred_data[response], pred_data[f'predicted_{response}']))



# Run the Model with Best Parameters and Save Test Predictions
data = league_data_loader(seasons=list(range(2010, 2020)))
features = ['season_type', 'date_num']
best_model, best_params = hyperparameter_tuning(data, list(range(2010, 2015)), list(range(2015, 2016)), list(range(2016, 2020)), features, 'league_avg_fg3a_fga', param_grid=None)
torch.save(best_model.state_dict(), 'best_model.pth')
run_and_save_predictions(data, list(range(2010, 2015)), list(range(2015, 2016)), list(range(2016, 2020)), features, 'league_avg_fg3a_fga', best_params)

