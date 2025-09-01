# task_2_preprocess_data.py

import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt

def preprocess_data(input_path, output_path):
    """
    Applies filtering, noise reduction, and feature engineering
    to the cleaned data.
    """
    print("\nPreprocessing data...")
    players_df = pd.read_csv(f"{input_path}/players_df.csv")
    ball_df = pd.read_csv(f"{input_path}/ball_df.csv")

    # Define pitch boundaries
    PITCH_X_MIN, PITCH_X_MAX = -52.5, 52.5
    PITCH_Y_MIN, PITCH_Y_MAX = -34, 34

    # Apply the boundary filter
    initial_player_rows = len(players_df)
    players_df = players_df[
        (players_df['Pitch_x'] >= PITCH_X_MIN) & (players_df['Pitch_x'] <= PITCH_X_MAX) &
        (players_df['Pitch_y'] >= PITCH_Y_MIN) & (players_df['Pitch_y'] <= PITCH_Y_MAX)
    ].copy()
    print(f"Removed {initial_player_rows - len(players_df)} player data points outside pitch boundaries.")

    # Apply Butterworth Filter for noise reduction
    fs = 10.0
    fc = 1.2
    order = 2
    Wn = fc / (fs / 2)
    b, a = butter(order, Wn, btype='low')

    players_df['x_smooth'] = players_df.groupby('participation_id')['Pitch_x'].transform(lambda x: filtfilt(b, a, x, padlen=len(x)-1))
    players_df['y_smooth'] = players_df.groupby('participation_id')['Pitch_y'].transform(lambda y: filtfilt(b, a, y, padlen=len(y)-1))

    # Feature Engineering: Calculate smoothed speed and acceleration
    def calculate_kinematics(player_data):
        delta_x = player_data['x_smooth'].diff()
        delta_y = player_data['y_smooth'].diff()
        delta_time = player_data['Time (s)'].diff()
        distance = np.sqrt(delta_x**2 + delta_y**2)
        player_data['speed_smooth'] = distance / delta_time
        player_data['acceleration'] = player_data['speed_smooth'].diff() / delta_time
        return player_data.fillna(0)

    players_df = players_df.groupby('participation_id', group_keys=False).apply(calculate_kinematics)
    print("Preprocessing complete.")

    # Save the preprocessed data
    players_df.to_csv(f"{output_path}/players_preprocessed.csv", index=False)
    print(f"Preprocessed data saved to {output_path}")

if __name__ == "__main__":
    preprocess_data(
        input_path='F:/PD_task/task/pipeline/output',
        output_path='F:/PD_task/task/pipeline/output'
    )