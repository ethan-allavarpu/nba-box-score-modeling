import matplotlib.pyplot as plt
import pandas as pd
import ssl
import sys

sys.path.append("..")
from modeling.league_avg import preprocess_data

if __name__ == "__main__":
    ssl._create_default_https_context = ssl._create_unverified_context

    seasons = [2014, 2015]
    metric = "league_avg_fg3a_fga"
    league_game_df = preprocess_data(seasons)
    df = league_game_df.groupby("game_date")[metric].mean().reset_index()
    df = pd.merge(df, league_game_df[["game_date", "season"]], how="inner", on="game_date")
    colors = ["darkblue", "darkgreen"]
    for s, color in zip(df.season.unique(), colors):
        season_df = df[df.season == s]
        print(season_df.shape)
        plt.plot(season_df.game_date, season_df[metric], label=s, color=color)
    plt.xticks(rotation=45)
    plt.title("3-pt FGA / Total FGA over Two Seasons")
    plt.xlabel("Date")
    plt.ylabel("3-pt FGA / Total FGA")
    plt.legend(title="Season")
    plt.savefig("../plots/fga3-fga.png", dpi=300, bbox_inches="tight")
