import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

# Ensemble model imports:
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import RidgeCV

# 1. Load the cleaned data.
data_path = r"LOLDATA\cleaned_data.csv"  # adjust path/extension if needed
df = pd.read_csv(data_path)

print("Data preview:")
print(df.head())

# 2. Define features and targets.
# Here, the inputs are the team names (your team and the opponent) and the targets are kills, deaths, assists.
X = df[["teamname", "opponent_team"]]
y = df[["kills", "deaths", "assists"]]

# 3. Split the data into training (80%) and test (20%) sets.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

print("\nTraining set shapes:")
print("X_train:", X_train.shape, "y_train:", y_train.shape)
print("Test set shapes:")
print("X_test:", X_test.shape, "y_test:", y_test.shape)

# 4. Build a preprocessing pipeline.
# Since our inputs are categorical team names, we use OneHotEncoder.
preprocessor = ColumnTransformer(
    transformers=[
        ("onehot", OneHotEncoder(handle_unknown="ignore"), ["teamname", "opponent_team"])
    ]
)

# ===============================================
# Option 1: Using a Random Forest Regressor
# ===============================================
ensemble_model = RandomForestRegressor(n_estimators=200, random_state=42)

# ===============================================
# Option 2: Using a Stacking Regressor
# ===============================================
# Uncomment the following block to try stacking instead:
# base_estimators = [
#     ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
#     ('ridge', RidgeCV())
# ]
# ensemble_model = StackingRegressor(
#     estimators=base_estimators,
#     final_estimator=RidgeCV()
# )
# ===============================================

# 5. Create a pipeline that applies preprocessing then fits the ensemble model.
model_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", ensemble_model)
])

# 6. Train the ensemble model on the training data.
model_pipeline.fit(X_train, y_train)

# 7. Predict on the test set.
y_pred = model_pipeline.predict(X_test)

# 8. Evaluate performance using Mean Squared Error.
mse = mean_squared_error(y_test, y_pred)
print("\nTest set Mean Squared Error:", mse)

# (Optional) Display a few sample predictions versus actual values.
results = X_test.copy()
results["Actual_kills"] = y_test["kills"].values
results["Predicted_kills"] = y_pred[:, 0]
results["Actual_deaths"] = y_test["deaths"].values
results["Predicted_deaths"] = y_pred[:, 1]
results["Actual_assists"] = y_test["assists"].values
results["Predicted_assists"] = y_pred[:, 2]
print("\nSample predictions:")
print(results.head())
