import sportsdataverse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Function to preprocess data
def preprocess_data(seasons):
    """
    Load NBA player boxscore data, preprocess it to calculate league average points per minute.

    Args:
    seasons (list): List of seasons to load data for.

    Returns:
    DataFrame: Preprocessed NBA league game data.
    """
    nba_df = sportsdataverse.nba.load_nba_player_boxscore(seasons=seasons)
    nba_df = nba_df.to_pandas()

    # Remove entries with 0 minutes played
    nba_df = nba_df[nba_df['minutes'] > 0]
    nba_df['pts_per_min'] = nba_df['points'] / nba_df['minutes']

    # Aggregate data to league-game level
    league_game_df = nba_df.groupby(['game_date']).agg(
        total_pts=('points', 'sum'),
        total_min=('minutes', 'sum'),
        total_games=('game_id', 'nunique'),
        season_type=('season_type', 'first'),
        season=('season', 'first')
    ).reset_index()

    # Calculate league average points per minute (adjusting total minutes by factor of 10)
    league_game_df['league_avg_ppm'] = league_game_df['total_pts'] / (league_game_df['total_min'] / 10)
    league_game_df.sort_values('game_date', inplace=True)
    
    # Calculate the number of days since last game
    league_game_df["days_since_last_game"] = league_game_df["game_date"].diff().dt.days.fillna(130)
    
    # TODO: use spline on this to capture seasonality
    league_game_df['date_num'] = league_game_df.groupby(['season']).cumcount() + 1
    
    return league_game_df

def fit_and_summarize_arima(league_game_df):
    """
    Fit ARIMA model to the entire dataset and provide a summary, including ACF and PACF plots.

    Args:
    league_game_df (DataFrame): The preprocessed league game data.
    """
    # Fit the ARIMA model
    arima_model = ARIMA(
        endog=league_game_df['league_avg_ppm'], 
        exog=league_game_df[['days_since_last_game', 'season_type', 'date_num']], 
        order=(1, 0, 0)
    )
    arima_model_fit = arima_model.fit()

    # Print out the summary of the ARIMA model
    print(arima_model_fit.summary())

    # Plot ACF and PACF
    plt.figure(figsize=(14, 7))
    plot_acf(league_game_df['league_avg_ppm'], ax=plt.gca(), lags=40)
    plt.title('Autocorrelation Function (ACF)')
    plt.show()

    plt.figure(figsize=(14, 7))
    plot_pacf(league_game_df['league_avg_ppm'], ax=plt.gca(), lags=40, method='ywm')
    plt.title('Partial Autocorrelation Function (PACF)')
    plt.show()

# Function to fit ARIMA model and make one-day out-of-sample forecast
def fit_and_forecast(league_game_df, prediction_date):
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

    # Fit the ARIMA model
    arima_model = ARIMA(
        endog=fit_df['league_avg_ppm'], 
        exog=fit_df[['days_since_last_game', 'season_type', 'date_num']], 
        order=(1, 0, 0)
    )
    arima_model_fit = arima_model.fit()

    # Make forecast for the prediction_date
    exog_forecast = league_game_df[league_game_df['game_date'] == prediction_date][['days_since_last_game', 'season_type', 'date_num']]
    forecast = arima_model_fit.forecast(steps=1, exog=exog_forecast)

    return forecast.iloc[0]

# Main function to execute the script
def main():
    # Define the range of seasons to consider
    seasons = range(2014, 2019)

    # Preprocess the data
    league_game_df = preprocess_data(seasons)

    # Fit and summarize the ARIMA model
    fit_and_summarize_arima(league_game_df)

    # Store predictions in a dictionary
    predictions = {}

    # Iterate over each game date and predict the next one
    unique_dates = league_game_df[league_game_df['season'] >= 2015]['game_date'].unique()
    for i in range(len(unique_dates) - 1):
        prediction_date = unique_dates[i + 1]  # We predict the next day
        predictions[prediction_date] = fit_and_forecast(league_game_df, prediction_date)

    # Convert predictions to a DataFrame and display
    predictions_df = pd.DataFrame(list(predictions.items()), columns=['game_date', 'predicted_league_avg_ppm'])
    print(predictions_df)

    # looking at Pascal Siakam
    # lebron_df = nba_df.loc[nba_df.athlete_id == 3149673]
    # lebron_df.sort_values('game_date', inplace=True)

# Run the main function
if __name__ == "__main__":
    main()
