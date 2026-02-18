from tarfile import data_filter

import pandas as pd


def load_data(path):
    df = pd.read_csv(path)
    return df

def clean_data(df):
    if "Season" in df.columns:
        df = df[df["Season"]=="Summer"]

    df["Medal"] = df["Medal"].fillna("No Medal")

    df.drop_duplicates()

    df["Year"] = df["Year"].astype(int)

    return df

print("Done Successfully")