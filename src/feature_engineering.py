import pandas as pd

def add_lag_features(df):

    df = df.sort_values(["NOC", "Year"])

    df['Previous_medal'] = df.groupby("NOC")["Total_Medals"].shift(1)
    df['Previous_gold'] = df.groupby("NOC")["Gold"].shift(1)

    df["Rolling_3_avg_medal"] = (
        df.groupby("NOC")["Total_Medals"]
        .rolling(3)
        .mean()
        .reset_index(level=0, drop=True)
    )

    df["Medal_growth_rate"] = (
        (df["Total_Medals"] - df['Previous_medal'])
        / df['Previous_medal']
    ) * 100

    df = df.fillna(0)

    return df



def add_host_feature(df, raw_df):

    host_mapping = (
        raw_df.groupby(["Year", "City"])["NOC"]
        .agg(lambda x: x.value_counts().idxmax())
        .reset_index()
    )

    host_mapping = host_mapping[["Year", "NOC"]]
    host_mapping.rename(columns={"NOC": "Host_NOC"}, inplace=True)

    df = df.merge(host_mapping, on="Year", how="left")

    df["Host"] = (df["NOC"] == df["Host_NOC"]).astype(int)

    df.drop(columns=["Host_NOC"], inplace=True)

    return df

def add_quality_features(df):

    # Gold ratio (avoid division by zero)
    df["Gold_Ratio"] = df.apply(
        lambda row: row["Gold"] / row["Total_Medals"]
        if row["Total_Medals"] > 0 else 0,
        axis=1
    )

    return df

def add_participation_growth(df):

    df = df.sort_values(["NOC", "Year"])

    df["Prev_Athletes"] = df.groupby("NOC")["Total_Athletes"].shift(1)

    df["Participation_Growth"] = (
        (df["Total_Athletes"] - df["Prev_Athletes"])
        / df["Prev_Athletes"]
    )

    df["Participation_Growth"] = df["Participation_Growth"].fillna(0)
    df.drop(columns=["Prev_Athletes"], inplace=True)

    return df

def add_efficiency_stability(df):

    df["Rolling_3_Efficiency"] = (
        df.groupby("NOC")["Medal_Efficiency"]
        .rolling(3)
        .mean()
        .reset_index(level=0, drop=True)
    )

    df["Rolling_3_Efficiency"] = df["Rolling_3_Efficiency"].fillna(0)

    return df
