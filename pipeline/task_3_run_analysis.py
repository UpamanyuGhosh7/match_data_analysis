# task_3_run_analysis.py

import pandas as pd
import numpy as np

def run_analysis(input_path):
    """
    Performs tactical and individual performance analysis and prints leaderboards.
    """
    print("\nRunning analysis...")
    players_df = pd.read_csv(f"{input_path}/players_preprocessed.csv")

    # --- Tactical Analysis ---
    HIGH_SPEED_THRESHOLD = 5.5
    ATTACKING_THIRD_LINE = 105 / 6
    pressing_counts = players_df.groupby('participation_id').apply(lambda df: len(df[(df['speed_smooth'] > HIGH_SPEED_THRESHOLD) & (df['x_smooth'] > ATTACKING_THIRD_LINE)]))
    pressing_leaderboard = pressing_counts.reset_index(name='Pressing Actions').sort_values(by='Pressing Actions', ascending=False)
    print("\n## Pressing-Intensity (Top 5 Players)")
    print(pressing_leaderboard.head().to_markdown(index=False))

    SPRINT_THRESHOLD = 7.0
    PENALTY_BOX_X_MIN = 105 / 2 - 16.5
    
    # --- FIXED LINE ---
    # The .sum() is now inside the apply function to calculate the sum for each player.
    sprints_into_box = players_df.groupby('participation_id').apply(
        lambda df: ((df['speed_smooth'] > SPRINT_THRESHOLD) & (df['x_smooth'] > PENALTY_BOX_X_MIN) & ~((df['speed_smooth'] > SPRINT_THRESHOLD) & (df['x_smooth'] > PENALTY_BOX_X_MIN)).shift(1).fillna(False)).sum()
    )
    
    box_entry_leaderboard = sprints_into_box.reset_index(name='Sprints into Box').sort_values(by='Sprints into Box', ascending=False)
    print("\n## Sprints into the Penalty Box (Top 5)")
    print(box_entry_leaderboard.head().to_markdown(index=False))

    # --- New Leaderboards ---
    # Total Distance
    total_distances = players_df.groupby('participation_id').apply(lambda df: np.sqrt(df['x_smooth'].diff()**2 + df['y_smooth'].diff()**2).sum())
    distance_leaderboard_top5 = total_distances.reset_index(name='Total Distance (m)').sort_values(by='Total Distance (m)', ascending=False).head(5)
    print("\n\n" + "="*50)
    print("## Leaderboard: Top 5 Players by Total Distance Covered")
    print("="*50)
    print(distance_leaderboard_top5.to_markdown(index=False, floatfmt=".2f"))

    # Sprinting Distance
    ZONE_5_THRESHOLD_MS = 25 * (5 / 18)
    sprinting_distances = players_df.groupby('participation_id').apply(lambda df: np.sqrt(df['x_smooth'].diff()**2 + df['y_smooth'].diff()**2)[df['speed_smooth'] > ZONE_5_THRESHOLD_MS].sum())
    sprinting_leaderboard_top5 = sprinting_distances.reset_index(name='Sprinting Distance (m)').sort_values(by='Sprinting Distance (m)', ascending=False).head(5)
    print("\n" + "="*50)
    print("## Leaderboard: Top 5 Players by Sprinting Distance (Zone 5)")
    print("="*50)
    print(sprinting_leaderboard_top5.to_markdown(index=False, floatfmt=".2f"))

    # Top Speed
    top_speeds = players_df.groupby('participation_id')['speed_smooth'].max()
    speed_leaderboard_top5 = top_speeds.reset_index(name='Top Speed (m/s)').sort_values(by='Top Speed (m/s)', ascending=False).head(5)
    print("\n" + "="*50)
    print("## Leaderboard: Top 5 Players by Top Speed")
    print("="*50)
    print(speed_leaderboard_top5.to_markdown(index=False, floatfmt=".2f"))

    print("\nAnalysis complete.")

if __name__ == "__main__":
    run_analysis(input_path='F:/PD_task/task/pipeline/output')