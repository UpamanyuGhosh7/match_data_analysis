# task_4_generate_visualizations.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Arc, Rectangle, Circle
from scipy.signal import butter, filtfilt
import networkx as nx


def draw_pitch(ax, pitch_color='#228B22', line_color='white'):
    """
    Draws a football pitch on a matplotlib axes object.
    """
    ax.set_facecolor(pitch_color)
    
    # Pitch Outline & Halfway Line
    ax.plot([0, 0], [-34, 34], color=line_color)
    ax.plot([-52.5, 52.5], [-34, -34], color=line_color)
    ax.plot([-52.5, 52.5], [34, 34], color=line_color)
    ax.plot([-52.5, -52.5], [-34, 34], color=line_color)
    ax.plot([52.5, 52.5], [-34, 34], color=line_color)
    
    # Center Circle
    center_circle = Circle((0.0, 0.0), 9.15, edgecolor=line_color, facecolor='none', lw=2)
    ax.add_patch(center_circle)
    center_dot = Circle((0.0, 0.0), 0.5, color=line_color)
    ax.add_patch(center_dot)
    
    # Penalty Areas
    ax.add_patch(Rectangle((-52.5, -20.15), 16.5, 40.3, edgecolor=line_color, facecolor='none'))
    ax.add_patch(Rectangle((36, -20.15), 16.5, 40.3, edgecolor=line_color, facecolor='none'))
    
    # 6-yard Box
    ax.add_patch(Rectangle((-52.5, -9.16), 5.5, 18.32, edgecolor=line_color, facecolor='none'))
    ax.add_patch(Rectangle((47, -9.16), 5.5, 18.32, edgecolor=line_color, facecolor='none'))

    return ax

def generate_visualizations(input_path, output_path):
    """
    Generates and saves visualizations from the preprocessed data.
    """
    print("\nGenerating visualizations...")
    players_df = pd.read_csv(f"{input_path}/players_preprocessed.csv")
    ball_df = pd.read_csv(f"{input_path}/ball_df.csv")
    
    # --- Leaderboard Plots ---
    sns.set_theme(style="whitegrid")
    total_distances = players_df.groupby('participation_id').apply(lambda df: np.sqrt(df['x_smooth'].diff()**2 + df['y_smooth'].diff()**2).sum())
    distance_leaderboard_top5 = total_distances.reset_index(name='Total Distance (m)').sort_values(by='Total Distance (m)', ascending=False).head(5)
    
    ZONE_5_THRESHOLD_MS = 25 * (5 / 18)
    sprinting_distances = players_df.groupby('participation_id').apply(lambda df: np.sqrt(df['x_smooth'].diff()**2 + df['y_smooth'].diff()**2)[df['speed_smooth'] > ZONE_5_THRESHOLD_MS].sum())
    sprinting_leaderboard_top5 = sprinting_distances.reset_index(name='Sprinting Distance (m)').sort_values(by='Sprinting Distance (m)', ascending=False).head(5)
    
    top_speeds = players_df.groupby('participation_id')['speed_smooth'].max()
    speed_leaderboard_top5 = top_speeds.reset_index(name='Top Speed (m/s)').sort_values(by='Top Speed (m/s)', ascending=False).head(5)

    plt.figure(figsize=(10, 6))
    ax1 = sns.barplot(x='Total Distance (m)', y='participation_id', data=distance_leaderboard_top5, palette='viridis', orient='h')
    ax1.set_title('Top 5 Players by Total Distance Covered', fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{output_path}/total_distance_leaderboard.png")
    
    plt.figure(figsize=(10, 6))
    ax2 = sns.barplot(x='Sprinting Distance (m)', y='participation_id', data=sprinting_leaderboard_top5, palette='plasma', orient='h')
    ax2.set_title('Top 5 Players by Sprinting Distance (Zone 5)', fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{output_path}/sprinting_distance_leaderboard.png")

    plt.figure(figsize=(10, 6))
    ax3 = sns.barplot(x='Top Speed (m/s)', y='participation_id', data=speed_leaderboard_top5, palette='magma', orient='h')
    ax3.set_title('Top 5 Players by Top Speed', fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{output_path}/top_speed_leaderboard.png")
    print("Leaderboard plots saved.")

    # --- Team Positional Heatmap ---
    fig, ax = plt.subplots(figsize=(16, 10))
    plt.style.use('dark_background')
    ax = draw_pitch(ax)


    sns.kdeplot(
    x=players_df['x_smooth'],
    y=players_df['y_smooth'],
    fill=True,
    cmap="hot",
    n_levels=50,
    alpha=0.6,  
    ax=ax
    )

    ax.set_xlim(-55, 55)
    ax.set_ylim(-36, 36)
    ax.set_title('Team Positional Heatmap', fontsize=20, color='white')
    ax.set_xlabel('Pitch Length (m)', color='white')
    ax.set_ylabel('Pitch Width (m)', color='white')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(f"{output_path}/team_positional_heatmap.png")
    print("Team positional heatmap saved.")
    
    # --- Ball Position Heatmap ---
    fs = 10.0
    fc = 1.2
    order = 2
    Wn = fc / (fs / 2)
    b, a = butter(order, Wn, btype='low')
    ball_df['x_smooth'] = filtfilt(b, a, ball_df['Pitch_x'])
    ball_df['y_smooth'] = filtfilt(b, a, ball_df['Pitch_y'])

    fig, ax = plt.subplots(figsize=(16, 10))
    plt.style.use('dark_background')
    ax = draw_pitch(ax)
    
    sns.kdeplot(
        x=ball_df['x_smooth'],
        y=ball_df['y_smooth'],
        fill=True,
        cmap="YlOrRd",
        n_levels=50,
        alpha=0.65,
        ax=ax
    )

    ax.set_xlim(-55, 55)
    ax.set_ylim(-36, 36)
    ax.set_title('Ball Position Heatmap', fontsize=20, color='white')
    ax.set_xlabel('Pitch Length (m)', color='white')
    ax.set_ylabel('Pitch Width (m)', color='white')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(f"{output_path}/ball_position_heatmap.png")
    print("Ball position heatmap saved.")

    # --- Possession Heatmaps ---
    POSSESSION_THRESHOLD_M = 2.5
    merged_df = pd.merge_asof(
        players_df.sort_values('Time (s)'),
        ball_df[['Time (s)', 'x_smooth', 'y_smooth']].sort_values('Time (s)').rename(columns={'x_smooth': 'ball_x', 'y_smooth': 'ball_y'}),
        on='Time (s)',
        direction='nearest',
        tolerance=0.1
    ).dropna()
    merged_df['distance_to_ball'] = np.sqrt((merged_df['x_smooth'] - merged_df['ball_x'])**2 + (merged_df['y_smooth'] - merged_df['ball_y'])**2)
    min_dist_per_timestep = merged_df.groupby('Time (s)')['distance_to_ball'].min()
    in_possession_times = min_dist_per_timestep[min_dist_per_timestep < POSSESSION_THRESHOLD_M].index
    out_of_possession_times = min_dist_per_timestep[min_dist_per_timestep >= POSSESSION_THRESHOLD_M].index
    in_possession_df = players_df[players_df['Time (s)'].isin(in_possession_times)]
    out_of_possession_df = players_df[players_df['Time (s)'].isin(out_of_possession_times)]
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    plt.style.use('dark_background')
    
    ax1 = draw_pitch(axes[0])
    sns.kdeplot(
        x=in_possession_df['x_smooth'], y=in_possession_df['y_smooth'],
        fill=True, cmap="Greens", n_levels=50, alpha=0.7,
        ax=ax1
    )
    ax1.set_title('Team Shape: In Possession', fontsize=20, color='white')
    ax1.set_xticks([]); ax1.set_yticks([])

    ax2 = draw_pitch(axes[1])
    sns.kdeplot(
        x=out_of_possession_df['x_smooth'], y=out_of_possession_df['y_smooth'],
        fill=True, cmap="Reds", n_levels=50, alpha=0.7,
        ax=ax2
    )
    ax2.set_title('Team Shape: Out of Possession', fontsize=20, color='white')
    ax2.set_xticks([]); ax2.set_yticks([])

    plt.tight_layout()
    plt.savefig(f"{output_path}/possession_heatmaps.png")
    print("Possession heatmaps saved.")

    # --- Passing Network ---
    player_ids = players_df['participation_id'].unique()
    player_positions = players_df.pivot_table(index='Time (s)', columns='participation_id', values=['x_smooth', 'y_smooth'])
    player_positions.ffill(inplace=True)
    ball_positions = ball_df.set_index('Time (s)')[['x_smooth', 'y_smooth']]
    ball_positions.columns = pd.MultiIndex.from_tuples([('ball_x', ''), ('ball_y', '')])
    full_tracking_df = player_positions.join(ball_positions, how='inner')

    delta_x = ball_df['x_smooth'].diff()
    delta_y = ball_df['y_smooth'].diff()
    delta_time = ball_df['Time (s)'].diff()
    ball_df['speed_smooth'] = np.sqrt(delta_x**2 + delta_y**2) / delta_time
    ball_df.fillna(0, inplace=True)
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(ball_df['speed_smooth'], height=8, distance=15)
    passes = []
    for peak_idx in peaks:
        try:
            time_of_kick = ball_df.iloc[peak_idx]['Time (s)']
            window = full_tracking_df.loc[time_of_kick-1:time_of_kick+3]
            
            pre_kick_positions = window.loc[time_of_kick-0.5:time_of_kick-0.1]
            player_dist_before = {pid: np.mean(np.sqrt((pre_kick_positions[('x_smooth', pid)] - pre_kick_positions[('ball_x', '')])**2 + (pre_kick_positions[('y_smooth', pid)] - pre_kick_positions[('ball_y', '')])**2)) for pid in player_ids}
            sender = min(player_dist_before, key=player_dist_before.get)
            
            post_kick_positions = window.loc[time_of_kick+1:time_of_kick+2.5]
            player_dist_after = {pid: np.mean(np.sqrt((post_kick_positions[('x_smooth', pid)] - post_kick_positions[('ball_x', '')])**2 + (post_kick_positions[('y_smooth', pid)] - post_kick_positions[('ball_y', '')])**2)) for pid in player_ids}
            receiver = min(player_dist_after, key=player_dist_after.get)
            
            if sender != receiver:
                passes.append((sender, receiver))
        except (KeyError, IndexError):
            continue

    pass_counts = pd.DataFrame(passes, columns=['sender', 'receiver']).value_counts().reset_index(name='count')
    if not pass_counts.empty:
        G = nx.from_pandas_edgelist(pass_counts, 'sender', 'receiver', ['count'], create_using=nx.DiGraph())
        avg_positions = players_df.groupby('participation_id')[['x_smooth', 'y_smooth']].mean().to_dict('index')
        pos = {pid: (data['x_smooth'], data['y_smooth']) for pid, data in avg_positions.items()}
        fig, ax = plt.subplots(figsize=(16, 10))
        draw_pitch(ax, pitch_color='white', line_color='grey')
        nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=1200, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
        nx.draw_networkx_edges(G, pos, edge_color=pass_counts['count'], width=np.log1p(pass_counts['count']) * 2.5, arrowsize=20, edge_cmap=plt.cm.plasma, ax=ax)
        ax.set_title('Team Passing Network', fontsize=20)
        plt.tight_layout()
        plt.savefig(f"{output_path}/passing_network.png")
        print("Passing network saved.")
    else:
        print("No passes detected, skipping passing network visualization.")


if __name__ == "__main__":
    generate_visualizations(
        input_path='F:/PD_task/task/pipeline/output',
        output_path='F:/PD_task/task/pipeline/output'

    )
