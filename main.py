import pandas as pd
from src.data_cleaning import load_data, clean_data
from src.aggregation import country_year_table
from src.feature_engineering import (
    add_lag_features,
    add_host_feature,
    add_quality_features,
    add_participation_growth,
    add_efficiency_stability
)
from src.model_training import train_model

DATA_PATH = "data/raw/olympics_dataset.csv"

if not os.path.exists(DATA_PATH):
    url = "https://drive.google.com/file/d/10-QqP7aB5dPdM4rcEBK1Fd8QmF3SKVBX/view?usp=sharing"
    df = pd.read_csv(url)
else:
    df = pd.read_csv(DATA_PATH)

def main():
    # Load
    df = load_data(DATA_PATH)

    # Clean
    df = clean_data(df)

    # Aggregate
    country_year_df = country_year_table(df)
    country_year_df = add_lag_features(country_year_df)
    country_year_df = add_host_feature(country_year_df, df)
    country_year_df = add_quality_features(country_year_df)
    country_year_df = add_participation_growth(country_year_df)
    country_year_df = add_efficiency_stability(country_year_df)

    # Save processed dataset
    processed_path = "data/processed/country_year_aggregated.csv"
    country_year_df.to_csv("data/processed/country_year_aggregated.csv", index=False)

    print("Data aggregation complete.")
    print(country_year_df.head())

    train_model(processed_path)

if __name__ == "__main__":
    main()


