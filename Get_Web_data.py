import os
import time
import requests
import pandas as pd
from tqdm import tqdm

# Create a directory to store NCAA stats
data_dir = "ncaa_data"
os.makedirs(data_dir, exist_ok=True)

# Explicit NCAA stats URLs from the dropdown menu
STAT_URLS = {
    "Three-Point Attempts per Game": "https://stats.ncaa.org/rankings/national_ranking?academic_year=2025.0&division=1.0&ranking_period=151.0&sport_code=MBB&stat_seq=625.0",
    "Three-Point Percentage": "https://stats.ncaa.org/rankings/national_ranking?academic_year=2025.0&division=1.0&ranking_period=151.0&sport_code=MBB&stat_seq=152.0",
    "Turnover Margin": "https://stats.ncaa.org/rankings/national_ranking?academic_year=2025.0&division=1.0&ranking_period=151.0&sport_code=MBB&stat_seq=519.0",
    "Rebounds Per Game": "https://stats.ncaa.org/rankings/national_ranking?academic_year=2025.0&division=1.0&ranking_period=151.0&sport_code=MBB&stat_seq=932.0",
    "Fouls Per Game": "https://stats.ncaa.org/rankings/national_ranking?academic_year=2025.0&division=1.0&ranking_period=151.0&sport_code=MBB&stat_seq=286.0",
    "Assists Per Game": "https://stats.ncaa.org/rankings/national_ranking?academic_year=2025.0&division=1.0&ranking_period=151.0&sport_code=MBB&stat_seq=216.0",
    "Steals Per Game": "https://stats.ncaa.org/rankings/national_ranking?academic_year=2025.0&division=1.0&ranking_period=151.0&sport_code=MBB&stat_seq=215.0",
    "Bench Points Per Game": "https://stats.ncaa.org/rankings/national_ranking?academic_year=2025.0&division=1.0&ranking_period=151.0&sport_code=MBB&stat_seq=1284.0",
    "Blocks Per Game": "https://stats.ncaa.org/rankings/national_ranking?academic_year=2025.0&division=1.0&ranking_period=151.0&sport_code=MBB&stat_seq=214.0",
    "Assist to Turnover Ratio": "https://stats.ncaa.org/rankings/national_ranking?academic_year=2025.0&division=1.0&ranking_period=151.0&sport_code=MBB&stat_seq=474.0",
    "Fast Break Points": "https://stats.ncaa.org/rankings/national_ranking?academic_year=2025.0&division=1.0&ranking_period=151.0&sport_code=MBB&stat_seq=1285.0",
    "Free Throw Percentage": "https://stats.ncaa.org/rankings/national_ranking?academic_year=2025.0&division=1.0&ranking_period=151.0&sport_code=MBB&stat_seq=150.0",
    "Free Throws Made Per Game": "https://stats.ncaa.org/rankings/national_ranking?academic_year=2025.0&division=1.0&ranking_period=151.0&sport_code=MBB&stat_seq=633.0",
    "Field Goal Percentage": "https://stats.ncaa.org/rankings/national_ranking?academic_year=2025.0&division=1.0&ranking_period=151.0&sport_code=MBB&stat_seq=148.0",
    "Turnovers Forced Per Game": "https://stats.ncaa.org/rankings/national_ranking?academic_year=2025.0&division=1.0&ranking_period=151.0&sport_code=MBB&stat_seq=931.0",
    "Turnovers Per Game": "https://stats.ncaa.org/rankings/national_ranking?academic_year=2025.0&division=1.0&ranking_period=151.0&sport_code=MBB&stat_seq=217.0",
    "Rebound Margin": "https://stats.ncaa.org/rankings/national_ranking?academic_year=2025.0&division=1.0&ranking_period=151.0&sport_code=MBB&stat_seq=151.0",
    "Defensive Rebounds Per Game": "https://stats.ncaa.org/rankings/national_ranking?academic_year=2025.0&division=1.0&ranking_period=151.0&sport_code=MBB&stat_seq=859.0",
    "Offensive Rebounds Per Game": "https://stats.ncaa.org/rankings/national_ranking?academic_year=2025.0&division=1.0&ranking_period=151.0&sport_code=MBB&stat_seq=857.0",
    "Team Win Percentage": "https://stats.ncaa.org/rankings/national_ranking?academic_year=2025.0&division=1.0&ranking_period=151.0&sport_code=MBB&stat_seq=168.0",
    "Effective Field Goal Percentage": "https://stats.ncaa.org/rankings/national_ranking?academic_year=2025.0&division=1.0&ranking_period=151.0&sport_code=MBB&stat_seq=1288.0",
    "Three Point Percentage Defense": "https://stats.ncaa.org/rankings/national_ranking?academic_year=2025.0&division=1.0&ranking_period=151.0&sport_code=MBB&stat_seq=518.0",
    "Scoring Offense": "https://stats.ncaa.org/rankings/national_ranking?academic_year=2025.0&division=1.0&ranking_period=151.0&sport_code=MBB&stat_seq=145.0"


}

# Headers to mimic a real browser request
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}


# Function to download and parse NCAA stats
def download_ncaa_stats(stat_urls):
    all_data = {}

    for stat_name, url in tqdm(stat_urls.items(), desc="Downloading NCAA Stats"):
        print(f"Fetching {stat_name} from {url}...")
        response = requests.get(url, headers=HEADERS)
        if response.status_code != 200:
            print(f"Failed to download {stat_name}")
            continue

        # Convert response text into a dataframe
        try:
            df = pd.read_html(response.text)[0]  # Extract first table

            # Ensure 'Team' column is unique
            df = df.loc[:, ~df.columns.duplicated()].copy()

            # Keep only relevant columns
            df = df[["Team"] + [col for col in df.columns if col not in ["GM", "W-L"]]]

            file_path = os.path.join(data_dir, f"{stat_name.replace(' ', '_')}.csv")
            df.to_csv(file_path, index=False)
            all_data[stat_name] = df
            print(f"Saved {stat_name} to {file_path}")
        except Exception as e:
            print(f"Error parsing {stat_name}: {e}")

        time.sleep(2)  # Avoid overwhelming the server

    return all_data


# Step 2: Merge Datasets into a Single CSV
def merge_ncaa_data(all_data):
    if not all_data:
        print("No data available to merge.")
        return

    # Merge multiple stat tables by team name
    merged_df = None
    for stat_name, df in all_data.items():
        if "Team" in df.columns:
            df = df.loc[:, ~df.columns.duplicated()].copy()
            df = df.rename(columns={df.columns[1]: stat_name})  # Rename second column with stat name
            if merged_df is None:
                merged_df = df
            else:
                merged_df = merged_df.merge(df, on="Team", how="left", suffixes=("", f"_{stat_name}"))

    # Save merged dataset
    merged_df.to_csv(os.path.join(data_dir, "ncaa_combined_stats.csv"), index=False)
    print("Merged dataset saved as ncaa_combined_stats.csv")


# Run the scraping process
all_data = download_ncaa_stats(STAT_URLS)
merge_ncaa_data(all_data) # merged files still contain redundancy due to .csv file stats + still have '_{original stat abbreviation}