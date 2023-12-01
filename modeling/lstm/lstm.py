import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from modeling.data_handling.data_loading import league_data_loader
import itertools

# Data Preparation Class
class SequenceDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence, ext_data, label = self.sequences[idx]
        return torch.FloatTensor(sequence), torch.FloatTensor(ext_data), torch.FloatTensor([label])

def create_inout_sequences(input_data, features, response):
    inout_seq = []
    input_data = input_data.reset_index(drop=True)
    for i in range(1, len(input_data)):
        seq_data = input_data.loc[:i-1, features + [response]].values

        ext_data = input_data.loc[i, features].values.astype(float)

        # The label is the 'league_avg_fg3a_fga' at time t
        label = input_data.loc[i, response]

        inout_seq.append((seq_data, ext_data, label))
    return inout_seq


def prepare_data(data, train_seasons, val_seasons, test_seasons, features, response):
    # Split the data into train, validation, and test sets
    train_data = data[data['season'].isin(train_seasons)]
    val_data = data[data['season'].isin(val_seasons)]
    test_data = data[data['season'].isin(test_seasons)]

    # sequences
    train_seq = create_inout_sequences(train_data, features, response)
    val_seq = create_inout_sequences(val_data, features, response)
    test_seq = create_inout_sequences(test_data, features, response)

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
    sequences, ext_data, labels = zip(*batch)
    lengths = [len(seq) for seq in sequences]
    sequences_padded = pad_sequence(sequences, batch_first=True)
    ext_data = torch.stack(ext_data)
    labels = torch.stack(labels)
    return sequences_padded, ext_data, labels, lengths

# LSTM Model Class with Pack Padded Sequence
class LSTMModel(nn.Module):
    def __init__(self, feature_size, hidden_size, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(feature_size + 1, hidden_size, num_layers, batch_first=True)
        # linear to includen the external data
        self.linear = nn.Linear(hidden_size + feature_size, 1)

    def forward(self, x, ext_data, lengths):
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_output, (ht, ct) = self.lstm(packed)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        # Selecting the output of the last time step for each sequence in the batch
        batch_indices = torch.arange(len(output))
        last_time_step_indices = [l - 1 for l in lengths]
        last_outputs = output[batch_indices, last_time_step_indices, :]

        # Concatenate the last output of the LSTM with the external data
        combined = torch.cat((last_outputs, ext_data), dim=1)

        # Pass the combined data through the linear layer for final prediction
        prediction = self.linear(combined)
        return prediction

# Training Function
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs):
    best_val_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for seq, ext_data, labels, lengths in train_loader:
            optimizer.zero_grad()
            y_pred = model(seq, ext_data, lengths)
            loss = criterion(y_pred, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        val_loss = evaluate_model(model, val_loader, criterion)
        print(f'Epoch {epoch+1}/{epochs}, Training Loss: {total_train_loss/len(train_loader)}, Validation Loss: {val_loss}')

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')

# Evaluation Function
def evaluate_model(model, loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for seq, ext_data, labels, lengths in loader:
            y_pred = model(seq, ext_data, lengths)
            loss = criterion(y_pred, labels)
            total_loss += loss.item()
    return total_loss / len(loader)

def hyperparameter_tuning(data, train_seasons, val_seasons, test_seasons, features, response, param_grid=None):
    # Default hyperparameters
    default_params = {
        'hidden_size': 128,
        'num_layers': 2,
        'lr': 5e-4,
        'batch_size': 1,
        'epochs': 10
    }

    if param_grid is None or not param_grid:
        # Use default hyperparameters
        param_grid = {k: [v] for k, v in default_params.items()}

    best_params = {}
    best_val_loss = float('inf')
    
    for params in itertools.product(*param_grid.values()):
        param_dict = dict(zip(param_grid.keys(), params))
        print("Testing with parameters:", param_dict)
        model = LSTMModel(feature_size=len(features), hidden_size=param_dict['hidden_size'], num_layers=param_dict['num_layers'])
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=param_dict['lr'], weight_decay=1e-5)

        train_dataset, val_dataset, test_dataset = prepare_data(data, train_seasons, val_seasons, test_seasons, features, response)
        train_loader, val_loader, test_dataset = create_data_loaders(train_dataset, val_dataset, test_dataset, batch_size=param_dict['batch_size'])

        train_model(model, train_loader, val_loader, criterion, optimizer, param_dict['epochs'])

        val_loss = evaluate_model(model, val_loader, criterion)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_params = param_dict

    print(f"Best Parameters: {best_params}, Best Validation Loss: {best_val_loss}")
    return best_params

# Function to Run the Model with Best Parameters and Save Test Predictions
def run_and_save_predictions(data, train_seasons, val_seasons, test_seasons, features, response, best_params):
    train_dataset, val_dataset, test_dataset = prepare_data(data, train_seasons, val_seasons, test_seasons, features, response)
    train_loader, val_loader, test_loader = create_data_loaders(train_dataset, val_dataset, test_dataset, batch_size=best_params['batch_size'])

    model = LSTMModel(input_size=len(features), hidden_size=best_params['hidden_size'], num_layers=best_params['num_layers'])
    model.load_state_dict(torch.load('best_model.pth'))
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=best_params['lr'], weight_decay=1e-5)

    train_model(model, train_loader, val_loader, criterion, optimizer, best_params['epochs'])

    # Save Test Predictions
    model.eval()
    test_predictions = []
    with torch.no_grad():
        for seq, ext_data, labels, lengths in test_loader:
            y_pred = model(seq, ext_data, lengths)
            test_predictions.extend(y_pred.tolist())

    pd.DataFrame(test_predictions, columns=['Predictions']).to_csv('test_predictions.csv', index=False)
    print("Test predictions saved to 'test_predictions.csv'.")


# Run the Model with Best Parameters and Save Test Predictions
data = league_data_loader(seasons=range(2014, 2020))
features = ['days_since_last_game', 'season_type', 'date_num']
best_params = hyperparameter_tuning(data, range(2012, 2016), range(2016, 2017), range(2017, 2020), features, 'league_avg_fg3a_fga', param_grid=None)
run_and_save_predictions(range(2012, 2016), range(2016, 2017), range(2017, 2020), features, 'league_avg_fg3a_fga')

