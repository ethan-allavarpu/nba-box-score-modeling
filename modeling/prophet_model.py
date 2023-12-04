from data_handling.data_loading import league_data_loader, player_data_loader
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def weighted_mse(true, pred, weights):
    return (weights * (true - pred) ** 2).sum() / weights.sum()

from prophet import Prophet

train_seasons=range(2010, 2015)
test_seasons=range(2016, 2020)
metric='fg3a_fga'
league_model=None

train_data = league_data_loader(seasons=list(train_seasons)).rename(columns={"game_date": "ds", "league_avg_fg3a_fga": "y"})

m = Prophet(weekly_seasonality=True, yearly_seasonality=True)
m.add_seasonality(name='monthly', period=30.5, fourier_order=10)
m = m.fit(train_data)

test_data = league_data_loader(seasons=list(test_seasons)).rename(columns={"game_date": "ds"})

prophet_preds = m.predict(test_data)

df = pd.merge(prophet_preds, test_data, on="ds")
print(weighted_mse(df.league_avg_fg3a_fga, df.yhat, df.fga))
print(mean_squared_error(df.league_avg_fg3a_fga, df.yhat))
print(r2_score(df.league_avg_fg3a_fga, df.yhat))

df.rename(columns={"ds": "game_date"}).to_csv(f'prophet_league_avg_{metric}_predictions.csv', index=False)
