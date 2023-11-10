import sportsdataverse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

def data_loader(seasons):
    nba_df = sportsdataverse.nba.load_nba_player_boxscore(seasons=seasons)
    nba_df = nba_df.to_pandas()

    # Remove entries with 0 minutes played
    nba_df = nba_df[nba_df['minutes'] > 0]
    # remove all-star games
    nba_df = nba_df[nba_df['team_id']<=30]
    nba_df['ppm'] = nba_df['points'] / nba_df['minutes']
    nba_df['fg3a_fga'] = nba_df['three_point_field_goals_attempted'] / nba_df['field_goals_attempted']
    return nba_df

# Function to preprocess data
def preprocess_data(seasons):
    """
    Load NBA player boxscore data, preprocess it to calculate league average points per minute.

    Args:
    seasons (list): List of seasons to load data for.

    Returns:
    DataFrame: Preprocessed NBA league game data.
    """
    nba_df = data_loader(seasons)
    # Aggregate data to league-game level
    league_game_df = nba_df.groupby(['game_date']).agg(
        total_pts=('points', 'sum'),
        fga=('field_goals_attempted', 'sum'),
        fg3a=('three_point_field_goals_attempted', 'sum'),
        total_min=('minutes', 'sum'),
        total_games=('game_id', 'nunique'),
        season_type=('season_type', 'first'),
        season=('season', 'first')
    ).reset_index()

    # Calculate league average points per minute
    league_game_df['league_avg_ppm'] = league_game_df['total_pts'] / league_game_df['total_min']
    league_game_df['league_avg_fg3a_fga'] = league_game_df['fg3a'] / (league_game_df['fga'])
    league_game_df.sort_values('game_date', inplace=True)
    
    # Calculate the number of days since last game
    league_game_df["days_since_last_game"] = league_game_df["game_date"].diff().dt.days.fillna(130)
    
    # TODO: use spline on this to capture seasonality
    league_game_df['date_num'] = league_game_df.groupby(['season']).cumcount() + 1
    
    return league_game_df

def fit_and_summarize_arima(league_game_df, metric):
    """
    Fit ARIMA model to the entire dataset and provide a summary, including ACF and PACF plots.

    Args:
    league_game_df (DataFrame): The preprocessed league game data.
    """

    # Plot ACF and PACF
    plt.figure(figsize=(14, 7))
    plot_acf(league_game_df[metric], ax=plt.gca(), lags=40)
    plt.title('Autocorrelation Function (ACF)')
    plt.savefig(f"acf_{metric}.png")


    plt.figure(figsize=(14, 7))
    plot_pacf(league_game_df[metric], ax=plt.gca(), lags=40, method='ywm')
    plt.title('Partial Autocorrelation Function (PACF)')
    plt.savefig(f"pacf_{metric}.png")

    # Fit the ARIMA model
    exog_df = league_game_df[['days_since_last_game', 'season_type']]
    # only keep columns that have more than 1 unique value
    exog_df = exog_df.loc[:, exog_df.nunique() > 1]
    arima_model = ARIMA(
        endog=league_game_df[metric], 
        exog=exog_df, 
        order=(5, 0, 0)
    )
    arima_model_fit = arima_model.fit()

    # Print out the summary of the ARIMA model
    print(arima_model_fit.summary())

# Function to fit ARIMA model and make one-day out-of-sample forecast
def fit_and_forecast(league_game_df, prediction_date, metric):
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
        order=(5, 0, 0)
    )
    arima_model_fit = arima_model.fit()

    # Make forecast for the prediction_date
    exog_forecast = league_game_df[league_game_df['game_date'] == prediction_date][['days_since_last_game', 'season_type']]
    forecast = arima_model_fit.forecast(steps=1, exog=exog_forecast)

    return forecast.iloc[0]

def main(metric='league_avg_fg3a_fga'):
    seasons = range(2014, 2019)

    # Preprocess the data
    league_game_df = preprocess_data(seasons)

    # Fit and summarize the ARIMA model
    fit_and_summarize_arima(league_game_df, metric=metric)

    # Store predictions in a dictionary
    predictions = {}

    unique_dates = league_game_df[league_game_df['season'] >= 2016]['game_date'].unique()
    for i in range(len(unique_dates) - 1):
        prediction_date = unique_dates[i + 1]  # We predict the next day
        predictions[prediction_date] = fit_and_forecast(league_game_df, prediction_date, metric=metric)

    predictions_df = pd.DataFrame(list(predictions.items()), columns=['game_date', f'predicted_{metric}'])
    
    predictions_df.to_csv(f'{metric}_predictions.csv', index=False)

def player_main(athlete_id, seasons, metric='fg3a_fga'):
    df = data_loader(seasons=seasons)
    # subset the data to the player
    df = df[df['athlete_id'] == athlete_id]
    athlete_name = df["athlete_display_name"].unique()[0]
    # sort by date
    df.sort_values('game_date', inplace=True)

    # Calculate the number of days since last game
    df["days_since_last_game"] = df["game_date"].diff().dt.days.fillna(130)

    # load the league average data
    league_game_df = pd.read_csv(f'league_avg_{metric}_predictions.csv')
    league_game_df['game_date'] = pd.to_datetime(league_game_df['game_date'])
    # merge the league average data with the player data
    df = pd.merge(df, league_game_df, on='game_date', how='inner')

    # delta between player and league average
    df[f"{athlete_name}_{metric}_delta"] = df[metric] - df[f'predicted_league_avg_{metric}']

    fit_and_summarize_arima(df, metric=f"{athlete_name}_{metric}_delta")



if __name__ == "__main__":
   main()
   # brook lopez
   player_main(3448, range(2014, 2019))
