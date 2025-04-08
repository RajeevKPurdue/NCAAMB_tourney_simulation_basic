import os
import pandas as pd
from glob import glob

# Define directory where CSV files are stored
data_dir = "ncaa_data"  # Change this if needed

# Get all CSV files in the directory
csv_files = glob(os.path.join(data_dir, "*.csv"))

# Initialize empty list to store cleaned dataframes
dataframes = []


def clean_dataset(df, filename):
    """ Cleans dataset by keeping all columns after 'Rank', ensuring unique columns, and splitting Team & Conference. """

    # Identify correct "Team" column and keep only columns after "Rank"
    rank_index = df.columns.get_loc("Rank") if "Rank" in df.columns else 0
    df = df.iloc[:, rank_index + 1:]  # Keep all columns after "Rank"

    # Ensure "Team" column exists
    if "Team.1" in df.columns:
        df = df.rename(columns={"Team.1": "Team"})
    if "Team" not in df.columns:
        print(f"Skipping {filename} - No 'Team' column found.")
        return None

    # Convert Team column to string and drop NaNs
    df["Team"] = df["Team"].astype(str).str.strip()
    df = df.dropna(subset=["Team"])

    # Split Team into "Team" and "Conference" only if format matches
    if df["Team"].str.contains(r"\(.+\)", regex=True).any():
        df[['Team', 'Conference']] = df['Team'].str.extract(r'(.+?) \((.+)\)')
        df["Conference"] = df["Conference"].str.replace("[()]", "", regex=True)  # Remove parentheses
    else:
        df["Conference"] = "Unknown"

    # Standardize column names (remove spaces, dashes)
    df = df.rename(columns=lambda x: x.strip().replace(" ", "_").replace("-", "_"))

    # Rename stat columns based on filename to prevent duplicates
    stat_name = os.path.basename(filename).replace(".csv", "")
    for col in df.columns[1:]:  # Avoid renaming Team & Conference
        if col not in ["Team", "Conference"]:
            df = df.rename(columns={col: f"{stat_name}_{col}"})

    return df


# Process each file
for file in csv_files:
    df = pd.read_csv(file)
    cleaned_df = clean_dataset(df, file)
    if cleaned_df is not None:
        dataframes.append(cleaned_df)

# Merge all dataframes on "Team" and "Conference", keeping only unique columns
merged_df = pd.concat(dataframes, axis=1)
merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]  # Remove duplicate columns

# Save merged dataframe
merged_df.to_csv(os.path.join(data_dir, "merged_ncaa_stats.csv"), index=False)

print("✅ Merged NCAA Stats saved as merged_ncaa_stats.csv")

#%%
import pandas as pd
# Create a directory to store NCAA stats
data_dir = "ncaa_data"
os.makedirs(data_dir, exist_ok=True)

# Load cleaned dataset
file_path = os.path.join(data_dir, "merged_ncaa_stats.csv")
df = pd.read_csv(file_path)

# -- df cleaning and listing selected features --
# Remove duplicate column names by renaming them appropriately
df = df.loc[:, ~df.columns.duplicated()]

# Rename columns with duplicate suffixes (e.g., "Offensive_Rebounds_Total.1")
df = df.rename(columns=lambda x: x.replace(".1", ""))

# Ensure all selected features exist in the dataset
available_features = [col for col in df.columns]

# Subset the dataset to only include relevant features and team/conference info
df_cleaned = df[["Team", "Conference"] + available_features]

# Isolating model training features
informative_features = [col for col in df_cleaned.columns if col not in ["Team", "Conference"]]

# Add seed variable to dataset
df_cleaned["Seed"] = df_cleaned["Team"].map(bracket_teams)

# Define new ratio-based features from "per_game" stats
per_game_stats = [col for col in df_cleaned.columns if "Per_Game" in col]

# Extract and print all columns that contain "Opp" (case-insensitive) to identify opponent stat naming patterns
matching_columns = [col for col in df_cleaned.columns if re.search(r"(_Opp|Opp|opp)", col, re.IGNORECASE)]

# Create ratio features for available opponent stats
stat_pairings = {}
for stat in df_cleaned.columns:
    if stat in matching_columns:  # Skip opponent stats, only work with team stats
        continue
    opp_stat = f"{stat}_Opp" if f"{stat}_Opp" in df_cleaned.columns else None
    if opp_stat and opp_stat in df_cleaned.columns:
        ratio_col = f"{stat}_Ratio"
        df_cleaned[ratio_col] = df_cleaned[stat].astype(float) / df_cleaned[opp_stat].astype(float)
        stat_pairings[ratio_col] = (stat, opp_stat)



# Convert features to numeric
df_cleaned[informative_features] = df_cleaned[informative_features].apply(pd.to_numeric, errors='coerce').fillna(0)
#%%
########## TRYING A CORRECT DATAFRAME CREATION
import os
import pandas as pd
import re

# Set directory containing the individual datasets
data_dir = "/Users/rajeevkumar/PycharmProjects/PythonProject4/ncaa_data"  # Change to your directory

# List all CSV files in the directory
files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]

# Initialize empty dataframe for merging
df_merged = None

for file in files:
    file_path = os.path.join(data_dir, file)

    # Read the dataset
    df = pd.read_csv(file_path)

    # Ensure first column is named "Team"
    df.rename(columns={df.columns[0]: "Team"}, inplace=True)
    df["Team"] = df["Team"].str.strip()

    # Extract only first column ("Team") and last stat column
    df = df.iloc[:, [0, -1]]

    # Save the original stat column name
    stat_colname_original = df.columns[-1]
    # Generate new column name
    new_stat_colname = f"{file.replace('.csv', '')}_{stat_colname_original}"
    df.rename(columns={stat_colname_original: new_stat_colname}, inplace=True)

    # Split "Team" into Team and Conference
    df[["Team", "Conference"]] = df["Team"].str.extract(r"^(.*?)\s*\((.*?)\)$")

    # Reorder columns: Team, Conference, Stat
    df = df[["Team", "Conference", new_stat_colname]]

    # Merge into main dataset
    if df_merged is None:
        df_merged = df
    else:
        df_merged = df_merged.merge(df, on=["Team", "Conference"], how="outer", suffixes=("", "_dup"))

# Drop any duplicate columns created during merge
df_merged = df_merged.loc[:, ~df_merged.columns.str.endswith("_dup")]

# Check post merge
df_merged = df_merged.merge(df, on=["Team", "Conference"], how="outer", suffixes=("", "_dup"))
print(f"  Merged shape after: {df_merged.shape}")

# Save the cleaned merged dataset
output_path = os.path.join(data_dir, "merged_ncaa_stats_cleaned_final.csv")
df_merged.to_csv(output_path, index=False)

print(f"Merged dataset saved to {output_path}")


#%%
# ---- Create Ratio-Based Features ----

# Identify opponent stats (features containing "Opp" or "_Opp_")
opponent_columns = [col for col in df_merged.columns if re.search(r"(_Opp|Opp|opp)", col, re.IGNORECASE)]

# Generate ratio-based features
for col in df_merged.columns:
    if col in opponent_columns:
        continue  # Skip opponent stats, focus on team stats

    opp_col = f"{col}_Opp" if f"{col}_Opp" in df_merged.columns else None
    if opp_col and opp_col in df_merged.columns:
        ratio_col = f"{col}_Ratio"
        df_merged[ratio_col] = df_merged[col] / df_merged[opp_col].replace(0, float("nan"))  # Avoid division by zero

# Save the cleaned merged dataset
output_path = os.path.join(data_dir, "merged_ncaa_stats_1.csv")
df_merged.to_csv(output_path, index=False)

print(f"Merged dataset saved to {output_path}")


#%%
import pandas as pd

# Load the cleaned NCAA dataset (replace with your actual file path)
file_path = "/Users/rajeevkumar/PycharmProjects/PythonProject4/ncaa_data/merged_ncaa_stats_cleaned_final.csv"
df_cleaned = pd.read_csv(file_path)

bracket_teams = {
    "Auburn": 1, "Alabama St.": 16, "Louisville": 8, "Creighton": 9, "Michigan": 5, "UC San Diego": 12,
    "Texas A&M": 4, "Yale": 13, "Ole Miss": 6, "North Carolina": 11, "Iowa St.": 3, "Lipscomb": 14,
    "Marquette": 7, "New Mexico": 10, "Michigan St.": 2, "Bryant": 15, "Duke": 1, "Mount St. Mary's": 16,
    "Mississippi St.": 8, "Baylor": 9, "Oregon": 5, "Liberty": 12, "Arizona": 4, "Akron": 13, "BYU": 6,
    "VCU": 11, "Wisconsin": 3, "Montana": 14, "Saint Mary's": 7, "Vanderbilt": 10, "Alabama": 2,
    "Robert Morris": 15, "Florida": 1, "Norfolk St.": 16, "UConn": 8, "Oklahoma": 9, "Memphis": 5,
    "Colorado St.": 12, "Maryland": 4, "Grand Canyon": 13, "Missouri": 6, "Drake": 11, "Texas Tech": 3,
    "UNCW": 14, "Kansas": 7, "Arkansas": 10, "St. John's": 2, "Omaha": 15, "Houston": 1,
    "SIUE": 16, "Gonzaga": 8, "Georgia": 9, "Clemson": 5, "McNeese": 12, "Purdue": 4,
    "High Point": 13, "Illinois": 6, "Xavier": 11, "Kentucky": 3, "Troy": 14, "UCLA": 7, "Utah St.": 10,
    "Tennessee": 2, "Wofford": 15
}

# Standardize team names by removing extra spaces and conference labels
df_cleaned["Team"] = df_cleaned["Team"].astype(str).str.strip()
df_cleaned["Team"] = df_cleaned["Team"].str.replace(r"\s*\(.*\)", "", regex=True)

# Map seeds to teams
df_cleaned["Seed"] = df_cleaned["Team"].map(bracket_teams)

# Filter out teams that are not in the tournament
df_tournament = df_cleaned.dropna(subset=["Seed"]).copy()

# Convert seed column to integer type
df_tournament["Seed"] = df_tournament["Seed"].astype(int)

# Save the filtered dataset
output_path = "filtered_ncaa_tournament_teams_2.csv"
df_tournament.to_csv(output_path, index=False)

print(f"Filtered dataset saved to: {output_path}")
#%%
########## ADJUSTING GOOD/BAD STATS TO POS/NEG, ADJUSTING STATS BY CONFERENCE USING MULTIPLIER

#"bad = Turnovers_Per_Game_TOPG, Fouls_Per_Game_DQ"

# Load dataset
df = pd.read_csv("/Users/rajeevkumar/PycharmProjects/PythonProject4/filtered_ncaa_tournament_teams_2.csv")

# --- Step 1: Reverse "bad" stats ---
bad_stats = ["Turnovers_Per_Game_TOPG", "Fouls_Per_Game_DQ"]
for col in bad_stats:
    if col in df.columns:
        df[col] = -1 * df[col]  # Reverse so higher is better

# --- Step 2: Normalize seed (higher number = stronger team) ---
if "Seed" in df.columns:
    df["Normalized_Seed"] = df["Seed"].max() - df["Seed"] + 1

# --- Step 3: One-hot encode conference (if exists) ---
#if "Conference" in df.columns:
    #df = pd.get_dummies(df, columns=["Conference"], prefix="Conf", drop_first=False)

# Print df head
print(df.head())

output_path = os.path.join('/Users/rajeevkumar/PycharmProjects/PythonProject4/ncaa_data', "filtered_ncaa_tournament_teams_adj_noconf_1.csv")

df.to_csv(output_path, index=False)