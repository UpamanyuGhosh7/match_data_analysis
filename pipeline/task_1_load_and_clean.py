# task_1_load_and_clean.py

import pandas as pd

def load_and_clean_data(input_path, output_path):
    """
    Loads the raw match data, performs initial cleaning and EDA,
    and saves the cleaned dataframes.
    """
    print("Loading and cleaning data...")
    # Load the dataset
    df = pd.read_csv(input_path)

    # Display basic info and the first few rows
    print("Initial DataFrame Info:")
    print(df.info())
    print("\nInitial DataFrame Head:")
    print(df.head())

    # Check for missing values
    print("\nMissing values per column:")
    print(df.isnull().sum())

    # Separate player data from ball data
    players_df = df[df['participation_id'] != 'ball'].copy()
    ball_df = df[df['participation_id'] == 'ball'].copy()

    print(f"\nTotal data points for players: {len(players_df)}")
    print(f"Total data points for ball: {len(ball_df)}")

    # Save the cleaned dataframes
    players_df.to_csv(f"{output_path}/players_df.csv", index=False)
    ball_df.to_csv(f"{output_path}/ball_df.csv", index=False)
    print(f"Cleaned data saved to {output_path}")

if __name__ == "__main__":
    load_and_clean_data(
        input_path='F:/PD_task/task/data/match_data.csv',
        output_path='F:/PD_task/task/pipeline/output'
    )