import matplotlib.pyplot as plt
import pandas as pd


if __name__ == "__main__":
    df = pd.read_csv("modeling/league_avg_fg3a_fga_predictions.csv")
    df_cnn = pd.read_csv("modeling/causal_cnn/cnn_test_predictions.csv")

    preds = df_cnn.Predictions.iloc[:-4].reset_index(drop=True)
    dates = pd.to_datetime(df.game_date.iloc[3:].reset_index(drop=True))
    true = df_cnn.league_avg_fg3a_fga.iloc[4:].reset_index(drop=True)
    weights = df_cnn.fga.iloc[4:].reset_index(drop=True)

    plt.figure(figsize=(4, 4), dpi=300)
    plt.scatter(true, preds, s=10, color="darkblue")
    plt.xlabel("Actual FG3A/FGA")
    plt.ylabel("Predicted FG3A/FGA")
    plt.savefig("plots/preds-actual.png", dpi=300, bbox_inches="tight")
    plt.clf()

    plt.figure(figsize=(4, 4), dpi=300)
    plt.plot(dates[dates < "2016-08-01"], (true - preds)[dates < "2016-08-01"], color="darkblue")
    plt.plot(
        dates[(dates < "2017-08-01") & (dates >= "2016-08-01")],
        (true - preds)[(dates < "2017-08-01") & (dates >= "2016-08-01")],
        color="darkblue"
    )
    plt.plot(
        dates[(dates < "2018-08-01") & (dates >= "2017-08-01")],
        (true - preds)[(dates < "2018-08-01") & (dates >= "2017-08-01")],
        color="darkblue"
    )
    plt.plot(dates[dates >= "2018-08-01"], (true - preds)[dates >= "2018-08-01"], color="darkblue")
    plt.xticks(["2016-01-01", "2017-01-01", "2018-01-01", "2019-01-01"], [*range(2016, 2020)])
    plt.xlabel("Date")
    plt.ylabel("Actual - Predicted")
    plt.savefig("plots/residuals.png", dpi=300, bbox_inches="tight")
    plt.clf()

    plt.figure(figsize=(4, 4), dpi=300)
    plt.scatter(weights, true - preds, s = 10, color="darkblue")
    plt.xlabel("Field Goals Attempted")
    plt.ylabel("Actual - Predicted")
    plt.savefig("plots/residuals-weights.png", dpi=300, bbox_inches="tight")
    plt.clf()
