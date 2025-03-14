#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to prepare test and training data for model evaluation
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Import from src modules
try:
    from src.utils.config import DATA_DIR
except ImportError:
    # Try direct import for when running from data_processing directory
    try:
        from utils.config import DATA_DIR
    except ImportError:
        logging.error("Failed to import required modules. Make sure you're running from the project root.")
        # Use default values
        DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")

def prepare_target_data(target_name, test_size=0.2, random_state=42):
    """
    Prepare training and testing data for a specific target
    
    Args:
        target_name (str): Name of the target variable (e.g., pts, reb, ast)
        test_size (float): Proportion of data to use for testing
        random_state (int): Random seed for reproducibility
        
    Returns:
        bool: True if successful, False otherwise
    """
    logging.info(f"Preparing data for target: {target_name}")
    
    # Load the processed data
    processed_data_files = [f for f in os.listdir(os.path.join(DATA_DIR, "processed")) 
                           if f.startswith("processed_nba_data_") and f.endswith(".csv")]
    
    if not processed_data_files:
        logging.error("No processed data files found")
        return False
    
    # Use the most recent file
    processed_data_file = sorted(processed_data_files)[-1]
    processed_data_path = os.path.join(DATA_DIR, "processed", processed_data_file)
    
    logging.info(f"Loading data from: {processed_data_path}")
    
    try:
        data = pd.read_csv(processed_data_path)
    except Exception as e:
        logging.error(f"Error loading data: {str(e)}")
        return False
    
    logging.info(f"Loaded data with {data.shape[0]} rows and {data.shape[1]} columns")
    
    # Check if target exists in the data
    if target_name not in data.columns:
        logging.error(f"Target column '{target_name}' not found in the data")
        return False
    
    # Select features (exclude some columns that are not useful for prediction)
    exclude_columns = [
        'gameID', 'game_date', 'longName', 'playerID', 'teamID', 'teamAbv', 
        'player_team_id', 'player_id', 'Season_Year', 'season_year', 'Player',
        'Tm', 'predominantly_low_minutes_player', 'low_minutes'
    ]
    
    # Also exclude any columns that have the same name as the target
    feature_columns = [col for col in data.columns 
                      if col != target_name and col not in exclude_columns]
    
    # Create feature matrix and target vector
    X = data[feature_columns]
    y = data[target_name]
    
    # Handle non-numeric columns
    numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
    X = X[numeric_cols]
    
    logging.info(f"Using {len(numeric_cols)} numeric features for prediction")
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Create training and testing dataframes
    train_data = X_train.copy()
    train_data[target_name] = y_train
    
    test_data = X_test.copy()
    test_data[target_name] = y_test
    
    # Save the training and testing data
    train_file = os.path.join(DATA_DIR, "processed", f"nba_{target_name}_train.csv")
    test_file = os.path.join(DATA_DIR, "processed", f"nba_{target_name}_test.csv")
    
    try:
        train_data.to_csv(train_file, index=False)
        logging.info(f"Saved training data to: {train_file}")
        
        test_data.to_csv(test_file, index=False)
        logging.info(f"Saved testing data to: {test_file}")
        
        return True
    except Exception as e:
        logging.error(f"Error saving data: {str(e)}")
        return False

def main():
    """Main function to prepare data for multiple targets"""
    # List of target variables to prepare data for
    targets = ['pts', 'reb', 'ast']
    
    success_count = 0
    for target in targets:
        if prepare_target_data(target):
            success_count += 1
    
    logging.info(f"Successfully prepared data for {success_count}/{len(targets)} targets")

if __name__ == "__main__":
    main()
