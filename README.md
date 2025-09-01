# Football Match Analysis Pipeline

This project provides a comprehensive data analysis pipeline for processing and visualizing football (soccer) match tracking data. It generates tactical insights, individual performance leaderboards, and a suite of visualizations from raw player and ball coordinates.

This repository offers two distinct methods to run the analysis:
1.  **A Modular Python Pipeline**: A series of sequential scripts for a structured, production-style workflow.
2.  **A Single Jupyter Notebook**: An interactive, self-contained environment for experimentation and step-by-step analysis.

## Prerequisites

* Python 3.x
* The required libraries, which include `pandas`, `numpy`, `scipy`, `matplotlib`, `seaborn`, and `networkx`. You can install them via pip:
    ```
    pip install pandas numpy scipy matplotlib seaborn networkx
    ```

## Dataset Overview

The dataset - `match_data.csv` consists of :

* `participation_id` – A unique identifier for each athlete.
* `time (s)` – Timestamp of the recorded data point. 
* `pitch_x and pitch_y` – Positional data on the pitch. 
* `speed (m/s)` – The athlete’s speed at the given time.

---

## Method 1: Modular Python Pipeline

This approach uses a series of Python scripts, each responsible for a specific stage of the data processing workflow. It is ideal for automated or production environments.

### Project Structure

The analysis pipeline is organized into five distinct scripts:

1.  **`task_1_load_and_clean.py`**: Loads the raw CSV data, separates it into player and ball dataframes, and performs initial cleaning.
2.  **`task_2_preprocess_data.py`**: Applies a Butterworth filter to smooth the tracking data, removes outliers, and engineers key kinematic features like speed and acceleration.
3.  **`task_3_run_analysis.py`**: Conducts the core analysis, calculating various performance metrics and generating leaderboards for tactical and physical performance.
4.  **`task_4_generate_visualizations.py`**: Creates a variety of visualizations from the processed data, including leaderboards, heatmaps, and a passing network.
5.  **`task_5_main.py`**: The main driver script that executes the entire pipeline in the correct sequence.

### How to Run the Pipeline

To run the full analysis, you simply need to execute the main script from your terminal.


#### Execution

1.  **Configure Paths**: Before running, open `task_5_main.py` and ensure the `input_data_path` and `output_folder` variables point to the correct locations on your system.
    ```python
    # Inside task_5_main.py
    input_data_path = 'path/to/your/match_data.csv'
    output_folder = 'path/to/your/output'
    ```
2.  **Run the main script**:
    ```
    python task_5_main.py
    ```
The script will execute all the steps sequentially and print progress updates to the console. All outputs, including cleaned data and visualizations, will be saved in the specified output folder.

### Pipeline Steps in Detail

#### Step 1: Load and Clean Data
* **Input**: `match_data.csv`
* **Action**: Loads the raw tracking data, separates it into `players_df.csv` and `ball_df.csv`, and handles initial data quality checks.
* **Output**: Cleaned `players_df.csv` and `ball_df.csv` in the output folder.

#### Step 2: Preprocess Data
* **Input**: `players_df.csv`, `ball_df.csv`
* **Action**: Applies a low-pass **Butterworth filter** to smooth coordinates and calculates kinematic features like smoothed speed and acceleration.
* **Output**: A `players_preprocessed.csv` file with the enriched data.

#### Step 3: Run Analysis
* **Input**: `players_preprocessed.csv`
* **Action**: Calculates tactical and physical performance metrics and prints the top 5 players for each metric to the console.
* **Metrics**: Pressing intensity, sprints into the penalty box, total distance, sprinting distance, and top speed.

#### Step 4: Generate Visualizations
* **Input**: `players_preprocessed.csv`, `ball_df.csv`
* **Action**: Generates and saves a series of plots and charts to the output folder.
* **Output Visualizations**: Leaderboard bar charts, positional heatmaps, and a passing network graph.

---

## Method 2: Jupyter Notebook Analysis

This approach uses a single Jupyter Notebook (`task.ipynb`) that contains the entire workflow, from data loading to visualization. It is ideal for interactive analysis and exploration.

### Analysis and Visualizations Overview

This pipeline produces the following key analyses and outputs:

* **Physical Performance Leaderboards**: Total Distance Covered, Sprinting Distance, Top Speed Achieved.
* **Tactical Metrics**: Pressing Intensity Actions, Sprints into the Penalty Box.
* **Positional Analysis Visualizations**: Team & Ball Positional Heatmaps, Team Shape Heatmaps (In/Out of Possession).
* **Passing Network Analysis**: A directed graph visualizing pass patterns between players.

### Setup and Dependencies

To run this analysis, you will need a Jupyter environment and the Python libraries mentioned in the prerequisites.


#### Data Requirements

The pipeline expects a single input file named `match_data.csv` with the following columns: `participation_id`, `Time (s)`, `Pitch_x`, `Pitch_y`, `Speed (m/s)`.

### Methodology and Workflow

The notebook follows a sequential data processing workflow:

1.  **Data Loading and Cleaning**: Loads `match_data.csv`, separates player and ball data, and performs an initial EDA.
2.  **Preprocessing and Feature Engineering**: Filters data to pitch boundaries, applies a **Butterworth low-pass filter** to smooth coordinates for noise reduction, and engineers new features (`speed_smooth`, `acceleration`).
3.  **Tactical and Performance Analysis**: Calculates leaderboards for top 5 players on key physical metrics and computes tactical metrics.
4.  **Visualization**: Generates bar charts for leaderboards, various heatmaps, and a passing network graph to visualize the results.

### Usage

To run the pipeline, simply open `task.ipynb` in a Jupyter environment and execute the cells in order from top to bottom. The notebook is self-contained and will print the analytical leaderboards and display all visualizations inline.