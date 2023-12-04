import pandas as pd
from modeling.data_handling.data_loading import league_data_loader, player_data_loader

from torch.utils.data import DataLoader, TensorDataset, Dataset
import torch
import torch.nn as nn


def get_hierarchical_data(seasons, athlete_names=[]):
    league_data = league_data_loader(seasons)
    player_data = player_data_loader(seasons)
    player_data = player_data.merge(league_data[['game_date', 'league_avg_fg3a_fga', 'date_num']], on='game_date', how='left')
    if len(athlete_names) > 0:
        player_data = player_data[player_data['athlete_display_name'].isin(athlete_names)]
    return league_data, player_data

def prepare_data(train_seasons, val_seasons, test_seasons, features, response, lag=4, athlete_names=[]):
    train_league_data, train_player_data = get_hierarchical_data(train_seasons, athlete_names)
    val_league_data, val_player_data = get_hierarchical_data(val_seasons, athlete_names)
    test_league_data, test_player_data = get_hierarchical_data(test_seasons, athlete_names)
    # weights
    train_league_data = create_weights(train_league_data, 'fga', train_seasons)
    val_league_data = create_weights(val_league_data, 'fga', val_seasons)
    test_league_data = create_weights(test_league_data, 'fga', test_seasons)
    train_player_data = create_weights(train_player_data, 'fga', train_seasons, groupby=['athlete_id'])
    val_player_data = create_weights(val_player_data, 'fga', val_seasons, groupby=['athlete_id'])
    test_player_data = create_weights(test_player_data, 'fga', test_seasons, groupby=['athlete_id'])

    # normalize features
    train_mean = train_league_data[features].mean()
    train_std = train_league_data[features].std()
    train_player_mean = train_player_data[features].mean()
    train_player_std = train_player_data[features].std()
    # normalize response
    train_response_mean = train_league_data[f'league_avg_{response}'].mean()
    train_response_std = train_league_data[f'league_avg_{response}'].std()
    train_response_player_mean = train_player_data[response].mean()
    train_response_player_std = train_player_data[response].std()

    # combine train, val, test
    full_league_df = pd.concat([train_league_data, val_league_data, test_league_data], axis=0)
    full_league_df = normalize_features(full_league_df, features, train_mean, train_std)
    # normalize response
    full_league_df = normalize_features(full_league_df, [f'league_avg_{response}'], train_response_mean, train_response_std)
    full_league_df = full_league_df.reset_index(drop=True).sort_values('game_date')

    # combine train, val, test
    full_player_df = pd.concat([train_player_data, val_player_data, test_player_data], axis=0)
    full_player_df = normalize_features(full_player_df, features, train_player_mean, train_player_std)
    # normalize response
    full_player_df = normalize_features(full_player_df, [response], train_response_player_mean, train_response_player_std)
    full_player_df = full_player_df.sort_values('game_date')

    train_player_sequences_all = []
    val_player_sequences_all = []
    test_player_sequences_all = []
    for athlete_name in full_player_df['athlete_display_name'].unique():
        player_df = full_player_df[full_player_df['athlete_display_name'] == athlete_name]
        player_df = player_df.reset_index(drop=True)
        train_start_index = player_df[player_df['season'].isin(train_seasons)].index.min()
        val_start_index = player_df[player_df['season'].isin(val_seasons)].index.min()
        test_start_index = player_df[player_df['season'].isin(test_seasons)].index.min()
        
        train_player_sequences = create_inout_sequences_player(full_league_df, player_df, features, response, train_start_index, lag, league_response_mean=train_response_mean, league_response_std=train_response_std,
        player_response_mean=train_response_player_mean, player_response_std=train_response_player_std)

        val_player_sequences = create_inout_sequences_player(full_league_df, player_df, features, response, val_start_index, lag, league_response_mean=train_response_mean, league_response_std=train_response_std,
        player_response_mean=train_response_player_mean, player_response_std=train_response_player_std)

        test_player_sequences = create_inout_sequences_player(full_league_df, player_df, features, response, test_start_index, lag,
        league_response_mean=train_response_mean, league_response_std=train_response_std, player_response_mean=train_response_player_mean, player_response_std=train_response_player_std)

        if athlete_name == full_player_df['athlete_display_name'].unique()[0]:
            train_player_sequences_all = train_player_sequences
            val_player_sequences_all = val_player_sequences
            test_player_sequences_all = test_player_sequences
        else:
            train_player_sequences_all += train_player_sequences
            val_player_sequences_all += val_player_sequences
            test_player_sequences_all += test_player_sequences
    
    # create datasets
    train_player_ds = PlayerSequenceDataset(train_player_sequences_all)
    val_player_ds = PlayerSequenceDataset(val_player_sequences_all)
    test_player_ds = PlayerSequenceDataset(test_player_sequences_all)

    # unnormalize for predictions
    full_player_df[features] = full_player_df[features] * train_player_std + train_player_mean
    full_player_df[response] = full_player_df[response] * train_response_player_std + train_response_player_mean
    full_league_df[features] = full_league_df[features] * train_std + train_mean
    full_league_df[f'league_avg_{response}'] = full_league_df[f'league_avg_{response}'] * train_response_std + train_response_mean
    return train_player_ds, val_player_ds, test_player_ds, full_player_df, full_league_df

def normalize_features(input_data, features, train_mean, train_std):
    input_data[features] = (input_data[features] - train_mean) / train_std
    return input_data

def create_weights(input_data, weight_feature, seasons, groupby=[]):
    if len(groupby) > 0:
        weight_sum = input_data.groupby(groupby)[weight_feature].transform('sum')
    else:
        weight_sum = input_data[weight_feature].sum()
    input_data['weights'] = input_data[weight_feature].values / weight_sum
    return input_data

class PlayerSequenceDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences
        # make sure league weight sum is same as player weight sum
        league_weight_sum = sum([league_fga_weight for _, _, _, _, _, _, _, league_fga_weight, _ in self.sequences])
        player_weight_sum = sum([fga_weight for _, _, fga_weight, _, _, _, _, _, _ in self.sequences])
        self.sequences = [(sequence, ext_data, fga_weight * league_weight_sum / player_weight_sum, label, 
        league_sequence, league_ext_data, league_label, league_fga_weight, player_name) for sequence, ext_data, fga_weight, label, league_sequence, league_ext_data, league_label, league_fga_weight, player_name in self.sequences]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence, ext_data, fga_weight, label, league_sequence, league_ext_data, league_label, league_fga_weight, player_name = self.sequences[idx]
        return (torch.FloatTensor(sequence), torch.FloatTensor(ext_data), torch.FloatTensor([fga_weight]), 
        torch.FloatTensor([label]), torch.FloatTensor(league_sequence), torch.FloatTensor(league_ext_data), 
        torch.FloatTensor([league_label]), torch.FloatTensor([league_fga_weight]), player_name)

def create_data_loaders(train_dataset, val_dataset, test_dataset, batch_size):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

def create_inout_sequences_player(league_data, player_data, features, response, start_index, lag, league_response_mean, league_response_std, player_response_mean, player_response_std):
    inout_seq = []
    league_data = league_data.reset_index(drop=True)
    player_data = player_data.reset_index(drop=True)
    for i in range(start_index - lag, len(player_data)):
        end = i + lag
        if (i < 0) or (end >= len(player_data)):
            continue
        game_date = player_data.loc[end, 'game_date']
        player_name = player_data.loc[end, 'athlete_display_name']
        player_seq_data = player_data.loc[i : (end - 1), features + [response]].values.astype(float)
        player_ext_data = player_data.loc[end, features].values.astype(float)
        player_fga_weight = player_data.loc[end, "weights"]
        player_label = player_data.loc[end, response] * player_response_std + player_response_mean
        league_data_subset = league_data[league_data['game_date'] <= game_date]
        league_seq_data = league_data_subset.loc[i : (end - 1), features + [f'league_avg_{response}']].values.astype(float)
        league_ext_data = league_data_subset.loc[end, features].values.astype(float)
        league_fga_weight = league_data_subset.loc[end, "weights"]
        league_label = league_data_subset.loc[end, f'league_avg_{response}'] * league_response_std + league_response_mean
        inout_seq.append((player_seq_data, player_ext_data, player_fga_weight, player_label, 
        league_seq_data, league_ext_data, league_label, league_fga_weight, player_name))
    return inout_seq

class HierarchicalLSTM(nn.Module):
    def __init__(self, feature_size, hidden_size, num_layers, player_names):
        super(HierarchicalLSTM, self).__init__()
        # 3 LSTMS - one for league data, one for player data, one player specific lstm
        self.league_lstm = nn.LSTM(feature_size + 1, hidden_size, num_layers, batch_first=True)
        self.league_fc = nn.Linear(hidden_size + feature_size, 1)
        self.player_lstm = nn.LSTM(feature_size +1, hidden_size, num_layers, batch_first=True)
        self.player_fc = nn.Linear(hidden_size + feature_size, 1)
        for player_name in player_names:
            setattr(self, f'player_lstm_{player_name}', nn.LSTM(feature_size + 1, hidden_size, num_layers, batch_first=True))
            setattr(self, f'player_fc_{player_name}', nn.Linear(hidden_size + feature_size, 1))

    def forward(self, x, ext_data, league_x, league_ext_data, player_name):
        # last hidden state of league lstm
        league_out, _ = self.league_lstm(league_x)
        league_out = league_out[:, -1, :]
        # last hidden state of player lstm
        player_out, _ = self.player_lstm(x)
        player_out = player_out[:, -1, :]
        # player specific lstm
        player_specific_lstm = getattr(self, f'player_lstm_{player_name[0]}')
        player_specific_fc = getattr(self, f'player_fc_{player_name[0]}')
        player_specific_out, _ = player_specific_lstm(x)
        player_specific_out = player_specific_out[:, -1, :]
        # concatenate ext_data
        league_out = torch.cat((league_out, league_ext_data), 1)
        player_out = torch.cat((player_out, ext_data), 1)
        player_specific_out = torch.cat((player_specific_out, ext_data), 1)
        # fc layer
        league_out = self.league_fc(league_out)
        player_out = self.player_fc(player_out)
        player_specific_out = player_specific_fc(player_specific_out)
        # make sure output is positive
        return league_out, player_out + player_specific_out + league_out

# Training Function
def train_model(model, train_loader, val_loader, optimizer, epochs, player_weight):
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model = None
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_player_loss = 0
        train_league_loss = 0
        for batch_idx, (x, ext_data, fga_weight, label, league_x, league_ext_data, league_label, league_fga_weight, player_name) in enumerate(train_loader):
            optimizer.zero_grad()
            league_out, output = model(x, ext_data, league_x, league_ext_data, player_name)
            # player loss + league loss
            player_loss = torch.sum(fga_weight * (output - label)**2) * player_weight
            league_loss = torch.sum(league_fga_weight * (league_out - league_label)**2) * (1 - player_weight)
            loss = player_loss + league_loss
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_player_loss += player_loss.item()
            train_league_loss += league_loss.item()
        train_losses.append(train_loss)
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_idx, (x, ext_data, fga_weight, label, league_x, league_ext_data, league_label, league_fga_weight, player_name) in enumerate(val_loader):
                league_out, output = model(x, ext_data, league_x, league_ext_data, player_name)
                loss = torch.sum(fga_weight * (output - label)**2) * player_weight
                loss += torch.sum(league_fga_weight * (league_out - league_label)**2) * (1 - player_weight)
                val_loss += loss.item()
        val_losses.append(val_loss)
        print(f'Epoch: {epoch}, Train Loss: {train_losses[-1]}, Val Loss: {val_losses[-1]}')
        print(f'Player Loss: {train_player_loss}, League Loss: {train_league_loss}')
        if val_losses[-1] < best_val_loss:
            best_val_loss = val_losses[-1]
            best_model = model
    return train_losses, val_losses, best_model

def generate_predictions(model, test_loader, player_df, league_df, test_seasons, athlete_names):
    model.eval()
    league_preds = []
    predictions = {athlete: [] for athlete in athlete_names}
    with torch.no_grad():
        for batch_idx, (x, ext_data, fga_weight, label, league_x, league_ext_data, league_fga_weight, league_label, player_name) in enumerate(test_loader):
            league_out, output = model(x, ext_data, league_x, league_ext_data, player_name)
            league_preds.append(league_out)
            predictions[player_name[0]].append(output)
    test_player_data, test_league_data = (player_df[player_df['season'].isin(test_seasons)], league_df[league_df['season'].isin(test_seasons)])
    test_player_data = create_weights(test_player_data, 'fga', train_seasons, groupby=['athlete_id'])
    # merge preds athlete by athlete
    for athlete in predictions.keys():
        predictions[athlete] = torch.cat(predictions[athlete]).numpy()
        test_player_data.loc[test_player_data['athlete_display_name'] == athlete, 'predictions'] = predictions[athlete]
    # print weighted mse grouped by player
    test_player_data['weighted_mse'] = test_player_data['weights'] * (test_player_data['predictions'] - test_player_data['fg3a_fga'])**2
    print(test_player_data.groupby('athlete_display_name')['weighted_mse'].sum())
    # subtract mean predictions and add 2 * validation mean and subtract training mean
    test_player_data['league_predictions'] = [pred.item() for pred in league_preds]
    # group by game date mean predictions
    test_player_data['league_predictions'] = test_player_data.groupby('game_date')['league_predictions'].transform('mean')
    test_league_data = test_league_data.merge(test_player_data[['game_date', 'league_predictions']], on='game_date', how='inner')
    # create weights
    test_league_data = create_weights(test_league_data, 'fga', test_seasons)
    # weighted mse
    test_league_data['weighted_mse'] = test_league_data['weights'] * (test_league_data['league_predictions'] - test_league_data['league_avg_fg3a_fga'])**2
    print(test_league_data['weighted_mse'].sum())

    # to csv
    test_player_data.to_csv('test_player_data.csv')
    test_league_data.to_csv('test_league_data.csv')

def weighted_mse(true, pred, weights):
    return (weights * (true - pred) ** 2).sum() / weights.sum()


athlete_names = ["Anthony Davis", "Brook Lopez"]
features = ['season_type', 'date_num']
train_seasons = list(range(2015, 2017))
val_seasons = list(range(2017, 2018))
test_seasons = list(range(2018, 2020))
train_dataset, val_dataset, test_dataset, player_df, league_df = prepare_data(train_seasons, val_seasons, test_seasons, features, 'fg3a_fga', lag=4, athlete_names=athlete_names)

train_loader, val_loader, test_loader = create_data_loaders(train_dataset, val_dataset, test_dataset, batch_size=1)

model = HierarchicalLSTM(len(features), 16, 2, athlete_names)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
train_losses, val_losses, best_model = train_model(model, train_loader, val_loader, optimizer, epochs=100, player_weight=0.025)
torch.save(best_model.state_dict(), 'best_hierarchical_model.pth')
generate_predictions(best_model, test_loader, player_df, league_df, test_seasons, athlete_names=athlete_names)
