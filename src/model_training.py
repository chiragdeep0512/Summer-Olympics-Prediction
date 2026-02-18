import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import numpy as np



def train_model(data_path):

    df = pd.read_csv(data_path)

    # Remove infinity values
    df = df.replace([np.inf, -np.inf], 0)

    # Remove NaN values
    df = df.fillna(0)

    # Drop rows with missing lag features
    df = df.dropna()

    X = df[[
        "Previous_medal",
        "Rolling_3_avg_medal",
        "Medal_growth_rate",
        "Total_Athletes",
        "Total_Events",
        "Medal_Efficiency",
        "Host"
    ]]

    # Target
    y = df["Total_Medals"]

    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    df["Medal_growth_rate"] = df["Medal_growth_rate"].replace([np.inf, -np.inf], 0)
    df["Medal_growth_rate"] = df["Medal_growth_rate"].fillna(0)

    # Model
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=42
    )

    model.fit(X_train, y_train)

    import matplotlib.pyplot as plt

    # Feature Importance
    importances = model.feature_importances_
    feature_names = X_train.columns

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    print("\nFeature Importance:\n")
    print(importance_df)

    # Plot
    plt.figure()
    plt.bar(importance_df["Feature"], importance_df["Importance"])
    plt.xticks(rotation=45)
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.show()

    # Predictions
    y_pred = model.predict(X_test)

    # Evaluation
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("R2 Score:", r2_score(y_test, y_pred))
    import os

    # Create models folder if not exists
    os.makedirs("models", exist_ok=True)

    # Save model
    joblib.dump(model, "models/medal_prediction_model.pkl")

    print("Model saved successfully.")

    return model

