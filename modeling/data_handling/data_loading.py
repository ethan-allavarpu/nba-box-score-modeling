import sportsdataverse
import pandas as pd
import numpy as np

def player_data_loader(seasons):
    if min(seasons) < 2013:
        nba_df = sportsdataverse.nba.load_nba_player_boxscore(seasons=range(min(seasons), 2013)).to_pandas()
        nba_df_2013 = sportsdataverse.nba.load_nba_player_boxscore(seasons=range(2013, max(seasons)+1)).to_pandas()
        nba_df = pd.concat([nba_df, nba_df_2013])
    else:
        nba_df = sportsdataverse.nba.load_nba_player_boxscore(seasons=seasons)

    # Remove entries with 0 minutes played and 0 field goals attempted
    nba_df = nba_df[(nba_df['minutes'] > 0) & (nba_df['field_goals_attempted'] > 0)]
    # remove all-star games
    nba_df = nba_df[nba_df['team_id']<=30]
    nba_df['ppm'] = nba_df['points'] / nba_df['minutes']
    nba_df['fg3a_fga'] = nba_df['three_point_field_goals_attempted'] / nba_df['field_goals_attempted']
    nba_df['season_type'] = (nba_df['season_type'] == 3).astype(int)
    return nba_df

# Function to preprocess data
def league_data_loader(seasons):
    """
    Load NBA player boxscore data, preprocess it to calculate league average points per minute.

    Args:
    seasons (list): List of seasons to load data for.

    Returns:
    DataFrame: Preprocessed NBA league game data.
    """
    nba_df = player_data_loader(seasons)
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
    league_game_df["days_since_last_game"] = np.clip(league_game_df["days_since_last_game"], 0, 10)
    
    
    # TODO: use spline on this to capture seasonality
    league_game_df['date_num'] = (league_game_df.groupby(['season']).cumcount() + 1) / league_game_df.groupby(['season'])['game_date'].transform('count')
    
    return league_game_df