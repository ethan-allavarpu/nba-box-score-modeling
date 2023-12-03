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

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data_handling.data_loading import league_data_loader

np_utils.set_random_seed(2023)

data = league_data_loader(seasons=list(range(2010, 2020)))

# Split into training sequence, test sequence

def get_x_y(seq, lag):
	x = np.array(
		[seq[start:(start + lag)] for start in range(len(seq) - lag)]
    )
	y = np.array([seq[pos] for pos in range(lag, len(seq))])
	return x, y

train_seq = np.array(
	data[data.season.isin(list(range(2010,2015)))].league_avg_fg3a_fga.tolist()
)
# Same as AR model
lag = 4

x, y = get_x_y(train_seq, lag=lag)
x = x.reshape((x.shape[0], x.shape[1], 1))

# Basic model
model = Sequential()
model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(lag, 1)))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit(x, y, epochs=1000, verbose=0)

test_seq = np.array(
	data[data.season.isin(list(range(2016,2020)))].league_avg_fg3a_fga.tolist()
)
test_x, test_y = get_x_y(test_seq, lag)
test_x = test_x.reshape((test_x.shape[0], test_x.shape[1], 1))
yhats = model.predict(test_x)
print(mean_squared_error(yhats, test_y))
print(r2_score(y_true=test_y, y_pred=yhats))
