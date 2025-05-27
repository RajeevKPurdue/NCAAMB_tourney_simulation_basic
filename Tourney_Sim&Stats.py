import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier  # keep if needed
from sklearn.svm import SVC

# Extract tournament teams and their seeds from the PDF bracket
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


# Create a directory to store NCAA stats
data_dir = "ncaa_data"
os.makedirs(data_dir, exist_ok=True)

# Load cleaned dataset
#file_path = os.path.join(data_dir, "/Users/rajeevkumar/PycharmProjects/PythonProject4/ncaa_data/filtered_ncaa_tournament_teams_adj.csv") # new cleaned df with only tournament teams
file_path = "/Users/rajeevkumar/PycharmProjects/PythonProject4/ncaa_data/filtered_ncaa_tournament_teams_adj_noconf_1.csv"

df = pd.read_csv(file_path)

# celaning team special characters
# -- df quality check and listing selected features --
# Remove duplicate column names by renaming them appropriately
df_cleaned = df.loc[:, ~df.columns.duplicated()]

# Ensure all selected features exist in the dataset
#available_features = [col for col in df.columns ]

# Subset the dataset to only include relevant features and team/conference info
#df_cleaned = df[["Team"] + available_features]

# Keep 'Team' and 'Seed' safe
non_feature_cols = ["Team", "Seed"]

# Select only numeric features for modeling (exclude 'Team')
informative_features = [col for col in df_cleaned.columns if col not in non_feature_cols]

# Ensure all informative features are numeric
df_cleaned[informative_features] = df_cleaned[informative_features].apply(pd.to_numeric, errors='coerce').fillna(0)

# Clean the 'Team' column properly
df_cleaned["Team"] = df_cleaned["Team"].astype(str).str.strip()
#

# Ensure no duplicate column names
df_cleaned = df_cleaned.loc[:, ~df_cleaned.columns.duplicated()]

# Selecting the frequently used basketball features
#key_features = [
 #   "Turnover_Margin_Ratio", "Assist_to_Turnover_Ratio_Ratio",
 #   "Offensive_Efficiency", "Defensive_Efficiency", "Free_Throw_Rate", "Three_Point_Attempt_Rate",
 #   "Adjusted_Tempo", "Three_Point_Variability", "Opponent_EFG_Ratio", "Opponent_TO_Ratio", "Blocks_Per_Game_BKPG",
 #   "Rebound_Margin_REB_MAR", "Fast_Break_Points_PPG", "Free_Throws_Made_Per_Game_Avg", "Free_Throw_Percentage_FT%",
 #   "Turnovers_Forced_Per_Game_Avg", "Assists_Per_Game_APG", "Effective_Field_Goal_Percentage_G", "Effective_Field_Goal_Percentage_Pct",
 #   "Steals_Per_Game_STPG", "Offensive_Rebounds_Per_Game_RPG", "Scoring_Offense_PPG", "Bench_Points_Per_Game_PPG", "Field_Goal_Percentage_FG%",
 # ]


# handling errors and ensuring numeric values
df_cleaned[informative_features] = df_cleaned[informative_features].apply(pd.to_numeric, errors='coerce').fillna(0)

print("Sample Team names:", df_cleaned["Team"].unique()[:5])


# Standardize/scale Features
scaler = StandardScaler()
X = scaler.fit_transform(df_cleaned[informative_features])

# Define ensemble model training
X_train, X_test, y_train, y_test = train_test_split(X, np.random.randint(0, 2, size=len(X)), test_size=0.2, random_state=42)

# Define ensemble models
models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    #'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, random_state=42),
    #'Neural Network': MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=42),
    "Ridge Classifier": RidgeClassifier(),
    "SVM": SVC(probability=True, kernel="linear")  # can include if needed
}

# Train models
for name, model in models.items():
    model.fit(X_train, y_train)

# Add model predictions to the dataframe
#for name, model in models.items():
 #   team_win_probabilities = model.predict_proba(X)[:, 1]
 #   df_cleaned[name + "_Win_Prob"] = team_win_probabilities

def safe_predict_proba(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    else:  # For RidgeClassifier or others without proba
        return model.decision_function(X)  # or model.predict(X)

for name, model in models.items():
    team_win_probabilities = safe_predict_proba(model, X)
    df_cleaned[name + "_Win_Prob"] = team_win_probabilities

# Display top teams per model before the tournament starts
for model_name in models.keys():
    sort_col = model_name + "_Win_Prob"
    print(f"\nTop Teams by {model_name}:")
    print(df_cleaned[["Team", sort_col]].sort_values(by=sort_col, ascending=False))

# Display top teams per model
df_cleaned[["Team"] + [col for col in df_cleaned.columns if "Win_Prob" in col]].sort_values(by="Random Forest_Win_Prob", ascending=False)


# Tournament Bracket Order (Using Seed to Structure Rounds)
#tournament_order = sorted(bracket_teams.items(), key=lambda x: x[1])
# Corrected bracket order (Dictionary items were entered in correct initial matchups)
tournament_order = list(bracket_teams.keys())

""""
# Function to simulate matchups round by round
def predict_winner(team1, team2):
    #Simulates a matchup between two teams using the trained models
    team1_stats = df_cleaned[df_cleaned["Team"] == team1][informative_features]
    team2_stats = df_cleaned[df_cleaned["Team"] == team2][informative_features]

    if team1_stats.empty or team2_stats.empty:
        print(f"Skipping matchup: {team1} vs {team2} - Missing data")
        return team1 if not team1_stats.empty else team2  # Default to team with data

    # Convert stats to arrays for calculations
    team1_stats = team1_stats.iloc[0].values
    team2_stats = team2_stats.iloc[0].values

    matchup_features = team1_stats - team2_stats  # Compute matchup differences

    # Predict with all models and average their probabilities
    model_predictions = np.mean([model.predict_proba([matchup_features])[:, 1] for model in models.values()], axis=0)

    # Determine winner
    return team1 if model_predictions[0] > 0.5 else team2
"""

# Define weights manually (adjust as needed based on performance or intuition)
model_weights = {
    'Logistic Regression': 0.3,
    'Naive Bayes': 0.3,
    'Decision Tree': 0.05,
    'Random Forest': 0.05,
    'Ridge Classifier': 0.15,
    'SVM': 0.15
}

# Modify your prediction ensemble:
def predict_winner(team1, team2):
    team1_stats = df_cleaned[df_cleaned["Team"] == team1][informative_features]
    team2_stats = df_cleaned[df_cleaned["Team"] == team2][informative_features]

    if team1_stats.empty or team2_stats.empty:
        print(f"Skipping matchup: {team1} vs {team2} - Missing data")
        return team1 if not team1_stats.empty else team2

    team1_stats = team1_stats.iloc[0].values
    team2_stats = team2_stats.iloc[0].values
    matchup_features = team1_stats - team2_stats

    weighted_probs = []
    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba([matchup_features])[:, 1][0]
        elif hasattr(model, "decision_function"):
            decision = model.decision_function([matchup_features])
            prob = 1 / (1 + np.exp(-decision))[0]  # convert decision function to probability
        else:
            prob = model.predict([matchup_features])[0]  # fallback
        weight = model_weights.get(name, 0)
        weighted_probs.append(prob * weight)

    avg_weighted_prob = sum(weighted_probs)
    return team1 if avg_weighted_prob > 0.5 else team2



def simulate_tournament_per_model(model_name):
    print(f"\n🧪 Simulating Tournament Using {model_name} Only:")
    current_round = tournament_order[:]
    round_number = 1
    model = models[model_name]

    while len(current_round) > 1:
        next_round = []
        print(f"\nRound {round_number}:")

        for i in range(0, len(current_round), 2):
            t1, t2 = current_round[i], current_round[i + 1]
            s1 = df_cleaned[df_cleaned["Team"] == t1][informative_features].values[0]
            s2 = df_cleaned[df_cleaned["Team"] == t2][informative_features].values[0]
            diff = s1 - s2
            if hasattr(model, "predict_proba"):
                prob = model.predict_proba([diff])[:, 1][0]
            elif hasattr(model, "decision_function"):
                decision = model.decision_function([diff])
                prob = 1 / (1 + np.exp(-decision))[0]
            else:
                prob = model.predict([diff])[0]

            winner = t1 if prob > 0.5 else t2
            next_round.append(winner)
            print(f"{t1} vs {t2} ➝ {winner}")
        current_round = next_round
        round_number += 1

    print(f"\n🏆 {model_name} Champion: {current_round[0]}")

# Function to simulate the full tournament
def simulate_tournament():
    """ Runs a full tournament simulation following bracket order """
    #current_round = [team for team, seed in tournament_order]
    current_round = tournament_order[:]
    round_number = 1

    while len(current_round) > 1:
        print(f"\n🏀 Round {round_number} Matchups:")
        next_round = []

        for i in range(0, len(current_round), 2):
            if i + 1 >= len(current_round):
                next_round.append(current_round[i])  # Auto-advance odd team
                print(f"Auto-advancing: {current_round[i]}")
                continue

            winner = predict_winner(current_round[i], current_round[i + 1])
            next_round.append(winner)
            print(f"{current_round[i]} vs {current_round[i + 1]} ➝ Winner: {winner}")

        current_round = next_round
        round_number += 1

    print(f"\n🏆 Tournament Champion: {current_round[0]}")

# Individual model simulations
for model_name in models:
    simulate_tournament_per_model(model_name)


# Generate list for simulation input
tournament_teams = df_cleaned["Team"]   #.tolist()

# Run the Full Ensemble tournament simulation
#simulate_tournament(tournament_teams) - input was needed before
simulate_tournament()

####### SIMULATION BOOTSTRAPPING N = 1000 FOR NOW

from collections import defaultdict, Counter

def simulate_tournament_bootstrap(models, df_cleaned, informative_features, tournament_order, N=1000, model_weights=None):
    championship_counts = Counter()
    advancement_counts = defaultdict(lambda: [0]*6)  # rounds 1 to 6 (R64 to Champion)

    def weighted_prediction(t1, t2):
        row1 = df_cleaned[df_cleaned["Team"] == t1]
        row2 = df_cleaned[df_cleaned["Team"] == t2]
        if row1.empty or row2.empty:
            return t1 if not row1.empty else t2

        s1 = row1[informative_features].values[0]
        s2 = row2[informative_features].values[0]
        diff = s1 - s2

        win_prob = 0
        for name, model in models.items():
            if hasattr(model, "predict_proba"):
                prob = model.predict_proba([diff])[:, 1][0]
            elif hasattr(model, "decision_function"):
                decision = model.decision_function([diff])[0]
                prob = 1 / (1 + np.exp(-decision))  # Sigmoid
            else:
                prob = model.predict([diff])[0]  # Fallback

            weight = model_weights.get(name, 1.0)
            win_prob += prob * weight

        win_prob /= sum(model_weights.values())
        return t1 if np.random.rand() < win_prob else t2

    for sim in range(N):
        round_teams = tournament_order[:]
        round_num = 0

        while len(round_teams) > 1:
            next_round = []
            for i in range(0, len(round_teams), 2):
                if i + 1 >= len(round_teams):
                    winner = round_teams[i]
                else:
                    winner = weighted_prediction(round_teams[i], round_teams[i+1])
                next_round.append(winner)
                advancement_counts[winner][round_num] += 1
            round_teams = next_round
            round_num += 1

        # Final winner
        championship_counts[round_teams[0]] += 1
        advancement_counts[round_teams[0]][5] += 1

        if (sim + 1) % 100 == 0:
            print(f"Completed simulation {sim + 1}/{N}")

    return championship_counts, advancement_counts

N_SIMULATIONS = 1000

# Your model_weights should already be defined, otherwise set:
# model_weights = {name: 1 for name in models}

champs, advancement = simulate_tournament_bootstrap(
    models,
    df_cleaned,
    informative_features,
    tournament_order,
    N=N_SIMULATIONS,
    model_weights=model_weights
)

# Display most common champions
print("\n🏆 Most Frequent Champions:")
for team, count in champs.most_common(10):
    print(f"{team}: {count} wins ({count/N_SIMULATIONS:.1%})")

# Optional: show how often teams reach each round
print("\n📊 Team Advancement Summary (R64 to Champion):")
for team, rounds in advancement.items():
    print(f"{team}: {[f'{c/N_SIMULATIONS:.0%}' for c in rounds]}")
#%%

import matplotlib.pyplot as plt
import seaborn as sns

def plot_champion_distribution(championship_counts, N):
    top_teams = dict(championship_counts.most_common(15))  # Top 15
    teams = list(top_teams.keys())
    wins = [top_teams[team] for team in teams]

    plt.figure(figsize=(12, 6))
    plt.bar(teams, wins, color="skyblue")
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Number of Championships")
    plt.title(f"🏆 Most Frequent Champions over {N} Simulations")
    plt.tight_layout()
    plt.show()

plot_champion_distribution(champs, N_SIMULATIONS)


def plot_advancement_heatmap(advancement_counts, N):
    rounds = ["R64", "R32", "Sweet 16", "Elite 8", "Final 4", "Champ"]
    df_adv = pd.DataFrame(advancement_counts, index=rounds).T
    df_adv = df_adv.div(N)

    plt.figure(figsize=(12, 10))
    sns.heatmap(df_adv, annot=True, fmt=".0%", cmap="YlGnBu")
    plt.title("📊 Team Advancement Rates Across Rounds")
    plt.ylabel("Team")
    plt.xlabel("Tournament Round")
    plt.tight_layout()
    plt.show()

# Histogram
plot_champion_distribution(champs, N_SIMULATIONS)

#Seaborn Heatmap
plot_advancement_heatmap(advancement, N_SIMULATIONS)
