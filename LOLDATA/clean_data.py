import pandas as pd

# Load your CSV data (adjust the filename/path as needed)
df = pd.read_csv("LOLDATA/2025_LOL_RAW.csv")

# Keep gameid temporarily so we can create the opponent_team feature.
cols_to_keep = [
    "gameid",      # used for grouping and creating opponent_team
    "position",    # lane/role
    "champion",    # champion played
    "teamname",    # your team name (and later, used to derive the opponent)
    "gamelength",  # game duration
    "result",      # win/loss result
    "kills",       # target: kills
    "deaths",      # target: deaths
    "assists"      # target: assists
]

# Create a cleaned DataFrame with only the columns you care about
df_clean = df[cols_to_keep].copy()

# --- Create an "opponent_team" column ---
# Assuming each game (identified by gameid) has exactly two teams, we group by gameid
opponent_dict = {}
for game, group in df_clean.groupby("gameid"):
    teams = group["teamname"].unique()
    if len(teams) == 2:
        team1, team2 = teams
        opponent_dict[(game, team1)] = team2
        opponent_dict[(game, team2)] = team1
    else:
        # For games with not exactly two teams, assign None or handle appropriately.
        for team in teams:
            opponent_dict[(game, team)] = None

# Apply the mapping t create a new column 'opponent_team'
df_clean["opponent_team"] = df_clean.apply(
    lambda row: opponent_dict.get((row["gameid"], row["teamname"])),
    axis=1
)

# Now that we have created the opponent_team feature, drop the gameid column if it's no longer needed.
df_clean = df_clean.drop(columns=["gameid"])

# --- Final Check ---
print("Cleaned DataFrame columns:", df_clean.columns.tolist())
print(df_clean.head())

# Save the cleaned DataFrame as a new CSV file (without the index)
df_clean.to_csv("LOLDATA/cleaned_data.csv", index=False)

print("Cleaned CSV saved as 'cleaned_data.csv'.")
