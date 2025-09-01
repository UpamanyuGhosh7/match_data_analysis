# main.py

import os
from task_1_load_and_clean import load_and_clean_data
from task_2_preprocess_data import preprocess_data
from task_3_run_analysis import run_analysis
from task_4_generate_visualizations import generate_visualizations


def main():
    """
    Runs the entire data processing and analysis pipeline.
    """
    # Define file paths
    input_data_path = 'F:/PD_task/task/data/match_data.csv'
    output_folder = 'F:/PD_task/task/pipeline/output'
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    # --- Step 1: Load and Clean Data ---
    load_and_clean_data(input_path=input_data_path, output_path=output_folder)
    
    # --- Step 2: Preprocess Data ---
    preprocess_data(input_path=output_folder, output_path=output_folder)
    
    # --- Step 3: Run Analysis ---
    run_analysis(input_path=output_folder)
    
    # --- Step 4: Generate Visualizations ---
    generate_visualizations(input_path=output_folder, output_path=output_folder)
    
    print("\nPipeline finished successfully!")

if __name__ == "__main__":
    main()