import sys
import os
import pickle
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# Setup basic logging (only critical steps)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MODEL_FILE = "model_pipeline.pkl"
DATA_FILE = "full_data.csv"

def train_model():
    logging.info("Training model using data from %s...", DATA_FILE)
    # Load full combined data from your scraper
    data = pd.read_csv(DATA_FILE)
    data["match-datetime"] = pd.to_datetime(data["match-datetime"])
    data["team1-score"] = pd.to_numeric(data["team1-score"], errors="coerce")
    data["team2-score"] = pd.to_numeric(data["team2-score"], errors="coerce")
    data["k"] = pd.to_numeric(data["k"], errors="coerce")
    data = data.dropna(subset=["team1-score", "team2-score", "k"])
    
    # For training, use all matches older than 60 days.
    max_date = data["match-datetime"].max()
    cutoff_date = max_date - pd.Timedelta(days=60)
    train_data = data[data["match-datetime"] < cutoff_date].copy()
    logging.info("Training data contains %d records.", len(train_data))
    
    # Compute historical average kills per round (avg_kpr)
    train_data["total_rounds"] = train_data["team1-score"] + train_data["team2-score"]
    train_data = train_data[train_data["total_rounds"] > 0]
    train_data["match_kpr"] = train_data["k"] / train_data["total_rounds"]
    avg_kpr_df = train_data.groupby("player-name")["match_kpr"].mean().reset_index()
    avg_kpr_df.rename(columns={"match_kpr": "avg_kpr"}, inplace=True)
    train_data = pd.merge(train_data, avg_kpr_df, on="player-name", how="left")
    
    # Feature engineering: compute odds difference and is_favorite.
    train_data["odds_diff"] = abs(train_data["team1-odds"] - train_data["team2-odds"])
    def compute_is_favorite(row):
        if row["player-team"] == row["team1"]:
            return 1 if row["team1-odds"] < row["team2-odds"] else 0
        elif row["player-team"] == row["team2"]:
            return 1 if row["team2-odds"] < row["team1-odds"] else 0
        else:
            return 0
    train_data["is_favorite"] = train_data.apply(compute_is_favorite, axis=1)
    
    features = ["team1-odds", "team2-odds", "odds_diff", "is_favorite",
                "player-team", "team1", "team2", "avg_kpr"]
    target = "k"
    X_train = train_data[features]
    y_train = train_data[target].astype(float)
    
    # Compute recency sample weights so that more recent training games (closer to cutoff) have more weight.
    min_train_date = train_data["match-datetime"].min()
    denom_days = (cutoff_date - min_train_date).days
    if denom_days == 0:
        train_data["recency_weight"] = 1.0
    else:
        train_data["recency_weight"] = 1 + train_data["match-datetime"].apply(lambda d: (d - min_train_date).days) / denom_days
    sample_weights = train_data["recency_weight"]
    
    categorical_features = ["player-team", "team1", "team2"]
    numeric_features = [col for col in features if col not in categorical_features]
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ]
    )
    model_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    
    model_pipeline.fit(X_train, y_train, regressor__sample_weight=sample_weights)
    
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model_pipeline, f)
    logging.info("Model training completed and saved to %s.", MODEL_FILE)
    return model_pipeline

def load_model():
    if os.path.exists(MODEL_FILE):
        with open(MODEL_FILE, "rb") as f:
            model = pickle.load(f)
        logging.info("Loaded model from %s.", MODEL_FILE)
        return model
    else:
        logging.info("Model file not found, training new model...")
        return train_model()

def parse_input_line(line):
    """
    Parse an input CSV line for a kill prediction.
    Expected format (comma-separated):
      e.g. "Val: deLb,f0569908-999a-4459-8334-0b9c67f42ba3,ESPORTS,map_1_and_2_kills,33.0,over,+100,2.0"
      
    This function ignores the provided odds and line values.
    It extracts the player identifier and assumes defaults for the missing pre-match features.
    """
    tokens = line.strip().split(",")
    if len(tokens) < 4:
        raise ValueError("Input line does not have enough fields.")
    
    # First token: should be like "Val: deLb" or "LoL: Woody"
    player_field = tokens[0].strip()
    if player_field.startswith("Val:"):
        player_name = player_field.replace("Val:", "").strip()
        sport_id = "VALORANT"
    elif player_field.startswith("LoL:"):
        player_name = player_field.replace("LoL:", "").strip()
        sport_id = "LOL"
    else:
        player_name = player_field
        sport_id = tokens[2].strip()  # fallback
    
    # Fourth token should be a stat identifier; we check if it relates to kills.
    stat_name = tokens[3].strip().lower()
    if "kill" not in stat_name:
        raise ValueError("Input line stat is not related to kills.")
    
    # We ignore the odds provided in the line. Instead, we fill in default values.
    default_team1_odds = 1.50   # Default prematch odds for favorite team.
    default_team2_odds = 2.50   # Default prematch odds for underdog team.
    default_odds_diff = abs(default_team1_odds - default_team2_odds)
    default_is_favorite = 1     # Assume player's team is the favorite.
    
    # For team identifiers, we use the player's team as a placeholder.
    # In a real scenario, you'd want to input or look up actual matchup information.
    team1 = player_name  # placeholder
    team2 = "Opponent"   # placeholder
    
    # Default historical average kills per round (avg_kpr).
    default_avg_kpr = 0.5   # This is a placeholder value.
    
    # Build a DataFrame for prediction matching the model's feature order.
    features_dict = {
        "team1-odds": default_team1_odds,
        "team2-odds": default_team2_odds,
        "odds_diff": default_odds_diff,
        "is_favorite": default_is_favorite,
        "player-team": player_name,
        "team1": team1,
        "team2": team2,
        "avg_kpr": default_avg_kpr
    }
    return pd.DataFrame([features_dict])

def main():
    model = load_model()
    
    if len(sys.argv) > 1:
        input_line = sys.argv[1]
    else:
        input_line = input("Enter a prediction line (CSV format): ")
    
    logging.info("Parsing input line for prediction...")
    try:
        X_new = parse_input_line(input_line)
    except Exception as e:
        logging.error("Error parsing input: %s", e)
        return
    
    predicted_kills = model.predict(X_new)[0]
    print(f"Predicted number of kills: {predicted_kills:.2f}")

if __name__ == "__main__":
    main()
