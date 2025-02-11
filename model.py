import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# Set up basic logging for major steps.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logging.info("Starting model training process.")

# ==========================
# 1. Load and Prepare Data
# ==========================
logging.info("Loading full_data.csv ...")
data = pd.read_csv("full_data.csv")
data["match-datetime"] = pd.to_datetime(data["match-datetime"])
data["team1-score"] = pd.to_numeric(data["team1-score"], errors="coerce")
data["team2-score"] = pd.to_numeric(data["team2-score"], errors="coerce")
data["k"] = pd.to_numeric(data["k"], errors="coerce")
data = data.dropna(subset=["team1-score", "team2-score", "k"])
logging.info(f"Data loaded: {len(data)} records.")

# ==========================
# 2. Split Data into Training and Test Sets
# ==========================
max_date = data["match-datetime"].max()
cutoff_date = max_date - pd.Timedelta(days=60)
# Use all matches older than cutoff as training data, but sample a fraction of recent matches as test data.
recent_data = data[data["match-datetime"] >= cutoff_date].copy()
older_data = data[data["match-datetime"] < cutoff_date].copy()
recent_train, recent_test = train_test_split(recent_data, test_size=0.5, random_state=42)
train_data = pd.concat([older_data, recent_train], ignore_index=True)
test_data = recent_test.copy()
logging.info(f"Training set: {len(train_data)} records; Test set: {len(test_data)} records.")

# ==========================
# 3. Compute Historical Average Kills Per Round (avg_kpr)
# ==========================
logging.info("Computing historical average KPR from training data...")
train_data["total_rounds"] = train_data["team1-score"] + train_data["team2-score"]
train_data = train_data[train_data["total_rounds"] > 0]
train_data["match_kpr"] = train_data["k"] / train_data["total_rounds"]
avg_kpr_df = train_data.groupby("player-name")["match_kpr"].mean().reset_index()
avg_kpr_df.rename(columns={"match_kpr": "avg_kpr"}, inplace=True)
train_data = pd.merge(train_data, avg_kpr_df, on="player-name", how="left")
test_data = pd.merge(test_data, avg_kpr_df, on="player-name", how="left")
overall_avg_kpr = train_data["avg_kpr"].mean()
test_data["avg_kpr"] = test_data["avg_kpr"].fillna(overall_avg_kpr)
logging.info("Historical avg_kpr computed.")

# ==========================
# 4. Feature Engineering (Pre-Match Only)
# ==========================
# Compute odds difference and is_favorite.
for df in [train_data, test_data]:
    df["odds_diff"] = abs(df["team1-odds"] - df["team2-odds"])
    df["is_favorite"] = df.apply(
        lambda row: 1 if (row["player-team"] == row["team1"] and row["team1-odds"] < row["team2-odds"]) or
                         (row["player-team"] == row["team2"] and row["team2-odds"] < row["team1-odds"]) else 0,
        axis=1
    )
train_data = train_data.dropna(subset=["k"])
test_data = test_data.dropna(subset=["k"])

# ==========================
# 5. Select Features and Target
# ==========================
features = ["team1-odds", "team2-odds", "odds_diff", "is_favorite", "player-team", "team1", "team2", "avg_kpr"]
target = "k"
X_train = train_data[features]
y_train = train_data[target].astype(float)
X_test = test_data[features]
y_test = test_data[target].astype(float)
logging.info("Features selected.")

# ==========================
# 6. Compute Recency Sample Weights for Training Data
# ==========================
logging.info("Computing recency sample weights...")
min_train_date = train_data["match-datetime"].min()
denom_days = (cutoff_date - min_train_date).days
if denom_days == 0:
    train_data["recency_weight"] = 1.0
else:
    train_data["recency_weight"] = 1 + train_data["match-datetime"].apply(
        lambda d: (d - min_train_date).days
    ) / denom_days
sample_weights = train_data["recency_weight"]

# ==========================
# 7. Preprocessing & Model Pipeline
# ==========================
logging.info("Setting up preprocessing and model pipeline...")
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

# ==========================
# 8. Train the Model
# ==========================
logging.info("Training the model (this may take a while)...")
model_pipeline.fit(X_train, y_train, regressor__sample_weight=sample_weights)
logging.info("Model training completed.")

# ==========================
# 9. Evaluate the Model
# ==========================
logging.info("Evaluating the model on the test set...")
y_pred = model_pipeline.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
logging.info(f"Test MSE: {mse:.2f}; Test R2: {r2:.2f}")

logging.info("Model training and evaluation process completed.")
