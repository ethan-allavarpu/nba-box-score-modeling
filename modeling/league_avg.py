import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from data_handling.data_loading import league_data_loader, player_data_loader

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def fit_and_summarize_arima(league_game_df, metric, order, athlete="overall"):
    """
    Fit ARIMA model to the entire dataset and provide a summary, including ACF and PACF plots.

    Args:
    league_game_df (DataFrame): The preprocessed league game data.
    """

    # Plot ACF and PACF
    plt.figure(figsize=(7, 7))
    plot_acf(league_game_df[metric], ax=plt.gca(), lags=40)
    plt.title('Autocorrelation Function (ACF)')
    plt.savefig(f"plots/acf_{metric}_{athlete}.png")


    plt.figure(figsize=(7, 7))
    plot_pacf(league_game_df[metric], ax=plt.gca(), lags=40, method='ywm')
    plt.title('Partial Autocorrelation Function (PACF)')
    plt.savefig(f"plots/pacf_{metric}_{athlete}.png")

    # Fit the ARIMA model
    exog_df = league_game_df[['days_since_last_game', 'season_type']]
    # only keep columns that have more than 1 unique value
    exog_df = exog_df.loc[:, exog_df.nunique() > 1]
    arima_model = ARIMA(
        endog=league_game_df[metric], 
        exog=exog_df, 
        order=order
    )
    arima_model_fit = arima_model.fit()

    # Print out the summary of the ARIMA model
    print(arima_model_fit.summary())

# Function to fit ARIMA model and make one-day out-of-sample forecast
def fit_and_forecast(league_game_df, prediction_date, metric, order):
    """
    Fit ARIMA model to data up to a specified date and make a one-day out-of-sample forecast.

    Args:
    league_game_df (DataFrame): The preprocessed league game data.
    prediction_date (str or pd.Timestamp): The date for which to make the forecast.

    Returns:
    float: The forecasted league average points per minute for the given date.
    """
    # Use data up to the day before prediction_date for fitting
    fit_df = league_game_df[league_game_df['game_date'] < prediction_date]

    exog_df = fit_df[['days_since_last_game', 'season_type']]
    # only keep columns that have more than 1 unique value
    exog_df = exog_df.loc[:, exog_df.nunique() > 1]
    # Fit the ARIMA model
    arima_model = ARIMA(
        endog=fit_df[metric], 
        exog=exog_df, 
        order=order
    )
    arima_model_fit = arima_model.fit()

    # Make forecast for the prediction_date
    exog_forecast = league_game_df[league_game_df['game_date'] == prediction_date][['days_since_last_game', 'season_type']]
    exog_forecast = exog_forecast.loc[:, exog_df.columns]
    forecast = arima_model_fit.forecast(steps=1, exog=exog_forecast)

    return forecast.iloc[0]

def main(metric='league_avg_fg3a_fga', train_seasons=range(2010, 2016), test_seasons=range(2016, 2020)):

    # Preprocess the data
    league_game_df = league_data_loader(seasons=list(train_seasons) + list(test_seasons))

    # Fit and summarize the ARIMA model
    train_league_df = league_game_df[league_game_df['season'].isin(train_seasons)]
    fit_and_summarize_arima(train_league_df, metric=metric, order=(4, 1, 0))

    # Store predictions in a dictionary
    predictions = {}

    unique_dates = league_game_df[league_game_df['season'].isin(test_seasons)]['game_date'].unique()
    for i in range(len(unique_dates) - 1):
        prediction_date = unique_dates[i + 1]  # We predict the next day
        predictions[prediction_date] = fit_and_forecast(league_game_df, prediction_date, metric=metric, order=(4, 1, 0))

    predictions_df = pd.DataFrame(list(predictions.items()), columns=['game_date', f'predicted_{metric}'])
    predictions_df = predictions_df.merge(league_game_df[['game_date', metric, 'fga']], on='game_date', how='inner')
    
    predictions_df.to_csv(f'{metric}_predictions.csv', index=False)

def player_main(athlete_name,  order, train_seasons=range(2016, 2018), test_seasons=range(2018, 2020), metric='fg3a_fga', league_model=None):
    df = player_data_loader(seasons=list(train_seasons) + list(test_seasons))
    # subset the data to the player
    df = df[df['athlete_display_name'] == athlete_name]
    # sort by date
    df.sort_values('game_date', inplace=True)

    # Calculate the number of days since last game
    df["days_since_last_game"] = df["game_date"].diff().dt.days.fillna(130)

    # load the league average data
    league_game_df = pd.read_csv(f'league_avg_{metric}_predictions.csv')
    league_game_df['game_date'] = pd.to_datetime(league_game_df['game_date'])
    if league_model is None:
        # load the league average data
        league_game_df = pd.read_csv(f'league_avg_{metric}_predictions.csv')
        league_game_df['game_date'] = pd.to_datetime(league_game_df['game_date'])
        # merge the league average data with the player data
        df = pd.merge(df, league_game_df, on='game_date', how='inner')

        # delta between player and league average
        df[f"{athlete_name}_{metric}_delta"] = df[metric] - df[f'predicted_league_avg_{metric}']
        pred_col = f'predicted_league_avg_{metric}'
    elif league_model[0] == "Prophet":
        # load the league average data
        league_game_df = pd.read_csv(league_model[1])[["game_date", "yhat"]]
        league_game_df['game_date'] = pd.to_datetime(league_game_df['game_date'])
        # merge the league average data with the player data
        df = pd.merge(df, league_game_df, on='game_date', how='inner')

        # delta between player and league average
        df[f"{athlete_name}_{metric}_delta"] = df[metric] - df['yhat']
        pred_col = 'yhat'
    else:
        # load the league average data
        preds_df = pd.read_csv(league_model[1])
        preds_df = preds_df.iloc[:-4, :][["Predictions"]].reset_index(drop=True)
        preds_df['game_date'] = league_game_df.game_date.iloc[3:].reset_index(drop=True)
        league_game_df = preds_df
        # merge the league average data with the player data
        df = pd.merge(df, league_game_df, on='game_date', how='inner')

        # delta between player and league average
        df[f"{athlete_name}_{metric}_delta"] = df[metric] - df['Predictions']
        pred_col = "Predictions"

    print(df.columns)

    fit_and_summarize_arima(df, metric=f"{athlete_name}_{metric}_delta", order=order, athlete=athlete_name)

    # Now fit the model and make predictions
    predictions = {}
    unique_dates = df[df['season'].isin(test_seasons)]['game_date'].unique()
    for i in range(len(unique_dates) - 1):
        prediction_date = unique_dates[i + 1]  # We predict the next day
        predictions[prediction_date] = fit_and_forecast(df, prediction_date, metric=f"{athlete_name}_{metric}_delta", order=order)
    
    predictions_df = pd.DataFrame(list(predictions.items()), columns=['game_date', f'predicted_{athlete_name}_{metric}_delta'])
    predictions_df = predictions_df.merge(league_game_df, on='game_date', how='inner')
    predictions_df = predictions_df.merge(df[['game_date', metric]], on='game_date', how='inner')
    predictions_df[f"predicted_player_{metric}"] = predictions_df[pred_col] + predictions_df[f'predicted_{athlete_name}_{metric}_delta']
    predictions_df = predictions_df.merge(df[['game_date', metric, 'field_goals_attempted']].rename(columns={metric: f"{athlete_name}_{metric}"}), on='game_date', how='inner')
    
    if league_model is None:
        predictions_df.to_csv(f'{athlete_name}_{metric}_predictions.csv', index=False)
    else:
        predictions_df.to_csv(f'{athlete_name}_{metric}_predictions_{league_model[0]}.csv', index=False)



if __name__ == "__main__":
   # main()
   # brook lopez
   # player_main(athlete_name="Brook Lopez", order=(4, 0, 0))
   # player_main(athlete_name="Brook Lopez", order=(4, 0, 0), league_model=("Prophet", "prophet_league_avg_fg3a_fga_predictions.csv"))
   # Anthony Davis
   # player_main(athlete_name="Anthony Davis", order=(1, 0, 1), league_model=None)
   # player_main(athlete_name="Anthony Davis", order=(1, 0, 0), league_model=("CNN", "causal_cnn/cnn_test_predictions.csv"))
   # player_main(athlete_name="Anthony Davis", order=(1, 0, 1), league_model=("LSTM", "lstm/lstm_test_predictions.csv"))
   player_main(athlete_name="Anthony Davis", order=(1, 0, 1), league_model=("Prophet", "prophet_league_avg_fg3a_fga_predictions.csv"))
