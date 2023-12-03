from keras import utils as np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Conv1D

import numpy as np
import os
import pandas as pd

from sklearn.metrics import mean_squared_error, r2_score
import sys

import ssl

ssl._create_default_https_context = ssl._create_unverified_context

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from data_handling.data_loading import league_data_loader

np_utils.set_random_seed(2023)

data = league_data_loader(seasons=list(range(2010, 2020)))

def weighted_mse(true, pred, weights):
    return (weights * (true - pred) ** 2).sum() / weights.sum()

# Split into training sequence, test sequence


def get_x_y(seq, lag):
    x = np.array([seq[start : (start + lag)] for start in range(len(seq) - lag)])
    y = np.array([seq[pos] for pos in range(lag, len(seq))])
    return x, y


train_seq = np.array(
    data[data.season.isin(list(range(2010, 2015)))].league_avg_fg3a_fga.tolist()
)
val_seq = np.array(
    data[data.season.isin(list(range(2015, 2016)))].league_avg_fg3a_fga.tolist()
)
# Same as AR model
lag = 4

x, y = get_x_y(train_seq, lag=lag)
x = x.reshape((x.shape[0], x.shape[1], 1))

# Basic CNN model
causal_cnn = Sequential()
causal_cnn.add(
    Conv1D(filters=128, kernel_size=3, activation="relu", input_shape=(lag, 1))
)
causal_cnn.add(Flatten())
causal_cnn.add(Dense(64, activation="relu"))
causal_cnn.add(Dense(1))
causal_cnn.compile(optimizer="adam", loss="mse")
causal_cnn.fit(x, y, epochs=500, verbose=0)

test_seq = np.array(
    data[data.season.isin(list(range(2016, 2020)))].league_avg_fg3a_fga.tolist()
)
test_x, test_y = get_x_y(test_seq, lag)
test_x = test_x.reshape((test_x.shape[0], test_x.shape[1], 1))
yhats = causal_cnn.predict(test_x).squeeze()
print(mean_squared_error(yhats, test_y.squeeze()))
print(r2_score(y_true=test_y.squeeze(), y_pred=yhats))

weights = np.array(
    data[data.season.isin(list(range(2016, 2020)))].fga.iloc[4:].tolist()
).squeeze()

pd.DataFrame({"true": test_y, "preds": yhats, "weights": weights}).to_csv("keras_cnn_preds.csv", index=False)
print(weighted_mse(test_y, yhats, weights))
