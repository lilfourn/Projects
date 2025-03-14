"""
NBA Data Quality Module

This module provides functions for checking and fixing data quality issues in NBA statistics data.
It addresses specific problems like:

1. Zero or Negative Values: Fixes implausible statistics (like negative minutes or points)
2. NaN/NA Value Handling: Identifies and fixes values like "#N/A" that might be parsed as zeros
3. TOV_x vs TOV_y Resolution: Resolves inconsistencies between different turnover columns
4. Low-Minute Players: Treats players with minimal playing time appropriately, including per-36 normalization
5. Derived Statistics Validation: Checks and fixes anomalies in calculated and derived statistics

Usage:
  from src.data_processing.data_quality import run_all_quality_checks
  cleaned_df = run_all_quality_checks(df, min_minutes=10, check_derived=True)

Individual fixes can also be applied:
  from src.data_processing.data_quality import fix_invalid_values, resolve_turnover_columns
  df = fix_invalid_values(df)
  df = resolve_turnover_columns(df)
  
For derived statistics validation:
  from src.data_processing.data_quality_check_derived import check_derived_stats
  df = check_derived_stats(df)
"""

import pandas as pd
import numpy as np
import re
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def fix_invalid_values(df, copy=True):
    """
    Fix invalid values in a dataframe
    
    Args:
        df (pd.DataFrame): DataFrame to fix
        copy (bool): Whether to make a copy of the dataframe
    
    Returns:
        pd.DataFrame: Fixed DataFrame
    """
    if copy:
        df = df.copy()
    
    # Get numeric columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
    # Fix negative values for stats that can't be negative
    non_negative_stats = ['pts', 'reb', 'ast', 'stl', 'blk', 'TOV', 'TOV_x', 'TOV_y', 
                         'fgm', 'fga', 'tptfgm', 'tptfga', 'ftm', 'fta', 'OffReb', 
                         'DefReb', 'PF', 'mins', 'MP']
    
    # Intersect with columns that actually exist in the dataframe
    non_negative_stats = [col for col in non_negative_stats if col in df.columns]
    
    # Log negative values before fixing
    for col in non_negative_stats:
        neg_count = (df[col] < 0).sum()
        if neg_count > 0:
            logging.warning(f"Found {neg_count} negative values in {col}, replacing with 0")
            df[col] = df[col].clip(lower=0)
    
    # Fix percentage columns that exceed 100%
    pct_cols = ['fgp', 'tptfgp', 'ftp', 'FG%', '3P%', 'FT%', 'eFG%', 'TS%']
    pct_cols = [col for col in pct_cols if col in df.columns]
    
    for col in pct_cols:
        # Check if column is actually a percentage (values > 1)
        if df[col].max() > 1:
            invalid_count = (df[col] > 100).sum()
            if invalid_count > 0:
                logging.warning(f"Found {invalid_count} values > 100 in {col}, capping at 100")
                df[col] = df[col].clip(upper=100)
        else:
            # For percentages stored as decimals (0-1)
            invalid_count = (df[col] > 1).sum()
            if invalid_count > 0:
                logging.warning(f"Found {invalid_count} values > 1 in {col}, capping at 1")
                df[col] = df[col].clip(upper=1)
    
    # Check for outliers in minutes played
    if 'mins' in df.columns:
        # NBA games are 48 minutes, but with overtime can go higher
        # A reasonable upper limit might be 60-65 minutes
        outlier_count = (df['mins'] > 65).sum()
        if outlier_count > 0:
            logging.warning(f"Found {outlier_count} records with mins > 65, capping at 65")
            df['mins'] = df['mins'].clip(upper=65)
    
    return df

def fix_na_values(df, copy=True):
    """
    Fix NaN values that might be incorrectly parsed as zeros
    
    Args:
        df (pd.DataFrame): DataFrame to fix
        copy (bool): Whether to make a copy of the dataframe
    
    Returns:
        pd.DataFrame: Fixed DataFrame
    """
    if copy:
        df = df.copy()
    
    # Check for #N/A strings and convert to NaN
    # For all object columns, check for "#N/A" strings
    for col in df.select_dtypes(include=['object']).columns:
        na_mask = df[col].isin(["#N/A", "#N/A", "N/A", "#VALUE!"])
        na_count = na_mask.sum()
        if na_count > 0:
            logging.warning(f"Found {na_count} '#N/A' strings in {col}, converting to NaN")
            df.loc[na_mask, col] = np.nan
    
    # Get numeric columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
    # For detection of suspiciously high NaN->0 conversion
    nan_conversion_report = {}
    
    # Check columns with too many zeros, which might indicate NaN conversion
    for col in numeric_cols:
        zero_pct = (df[col] == 0).mean() * 100
        
        # If more than 30% of values are exactly 0, it could be suspicious
        # (this threshold might need adjustment based on domain knowledge)
        if zero_pct > 30:
            logging.warning(f"Column {col} has {zero_pct:.1f}% zeros, which might indicate incorrect NaN conversion")
            nan_conversion_report[col] = zero_pct
    
    # Log the report
    if nan_conversion_report:
        logging.warning("Columns with suspiciously high zero percentages (potential NaN conversion):")
        for col, pct in sorted(nan_conversion_report.items(), key=lambda x: x[1], reverse=True):
            logging.warning(f"  {col}: {pct:.1f}%")
    
    return df

def resolve_turnover_columns(df, copy=True):
    """
    Resolve inconsistencies between TOV_x and TOV_y columns
    
    Args:
        df (pd.DataFrame): DataFrame to fix
        copy (bool): Whether to make a copy of the dataframe
    
    Returns:
        pd.DataFrame: Fixed DataFrame
    """
    if copy:
        df = df.copy()
    
    # Check if both columns exist
    has_tov_x = 'TOV_x' in df.columns
    has_tov_y = 'TOV_y' in df.columns
    has_tov = 'TOV' in df.columns
    
    if has_tov_x and has_tov_y:
        # Count NaN values in each
        tov_x_na = df['TOV_x'].isna().sum()
        tov_y_na = df['TOV_y'].isna().sum()
        
        # Check for discrepancies where one is NA and the other isn't
        mismatch_mask = df['TOV_x'].isna() != df['TOV_y'].isna()
        mismatch_count = mismatch_count = mismatch_mask.sum()
        
        if mismatch_count > 0:
            logging.warning(f"Found {mismatch_count} rows where one TOV column has data and the other is NA")
            
            # Copy values from the non-NA column to the NA one
            x_from_y_mask = df['TOV_x'].isna() & df['TOV_y'].notna()
            y_from_x_mask = df['TOV_y'].isna() & df['TOV_x'].notna()
            
            if x_from_y_mask.sum() > 0:
                logging.info(f"Copying {x_from_y_mask.sum()} values from TOV_y to TOV_x")
                df.loc[x_from_y_mask, 'TOV_x'] = df.loc[x_from_y_mask, 'TOV_y']
            
            if y_from_x_mask.sum() > 0:
                logging.info(f"Copying {y_from_x_mask.sum()} values from TOV_x to TOV_y")
                df.loc[y_from_x_mask, 'TOV_y'] = df.loc[y_from_x_mask, 'TOV_x']
        
        # Check for discrepancies in non-NA values
        both_valid_mask = df['TOV_x'].notna() & df['TOV_y'].notna()
        if both_valid_mask.sum() > 0:
            diff = (df.loc[both_valid_mask, 'TOV_x'] - df.loc[both_valid_mask, 'TOV_y']).abs()
            has_diff = (diff > 0).sum()
            
            if has_diff > 0:
                mean_diff = diff[diff > 0].mean()
                logging.warning(f"Found {has_diff} rows with different values for TOV_x and TOV_y (mean diff: {mean_diff:.2f})")
                logging.info("Creating a unified 'TOV' column using TOV_x values when available, falling back to TOV_y")
                
                # Create a unified TOV column (prefer TOV_x, fall back to TOV_y)
                df['TOV'] = df['TOV_x'].combine_first(df['TOV_y'])
    elif has_tov_x:
        # Only TOV_x exists, rename it to TOV for consistency
        logging.info("Only TOV_x column found, creating unified TOV column")
        df['TOV'] = df['TOV_x']
    elif has_tov_y:
        # Only TOV_y exists, rename it to TOV for consistency
        logging.info("Only TOV_y column found, creating unified TOV column")
        df['TOV'] = df['TOV_y']
    
    return df

def handle_low_minute_players(df, min_minutes=10, copy=True):
    """
    Flag or filter players with minimal minutes
    
    Args:
        df (pd.DataFrame): DataFrame to process
        min_minutes (int): Minimum minutes threshold
        copy (bool): Whether to make a copy of the dataframe
    
    Returns:
        pd.DataFrame: DataFrame with additional columns for low minutes flags
    """
    if copy:
        df = df.copy()
    
    # Check if mins column exists
    if 'mins' not in df.columns:
        logging.warning("No 'mins' column found, cannot handle low minutes players")
        return df
    
    # Create a flag for low-minute appearances
    df['low_minutes'] = (df['mins'] < min_minutes).astype(int)
    
    # Calculate the percentage of low-minute games for each player
    if 'playerID' in df.columns:
        player_low_mins_pct = df.groupby('playerID')['low_minutes'].mean() * 100
        player_count = len(player_low_mins_pct)
        low_mins_players = (player_low_mins_pct > 50).sum()
        
        logging.info(f"{low_mins_players} out of {player_count} players have over 50% low-minute games")
        
        # Create a player-level flag for predominantly low-minute players
        low_mins_players_dict = (player_low_mins_pct > 50).to_dict()
        df['predominantly_low_minutes_player'] = df['playerID'].map(low_mins_players_dict).fillna(0).astype(int)
    
    # Create normalized stats for low-minute players (adjust the stats by minutes)
    # This helps prevent small-sample outliers, e.g., a player who got 2 blocks in 3 minutes
    for stat in ['pts', 'reb', 'ast', 'stl', 'blk', 'TOV', 'TOV_x', 'TOV_y']:
        if stat in df.columns:
            # Create per-36 minute values (industry standard for normalized stats)
            df[f'{stat}_per36'] = df[stat] * (36 / df['mins'].replace(0, np.nan))
            
            # Handle infinite values (divide by zero) and NaNs
            df[f'{stat}_per36'] = df[f'{stat}_per36'].replace([np.inf, -np.inf], np.nan)
            
            # Use reasonable caps based on NBA records
            caps = {
                'pts_per36': 60,  # Wilt's 100-pt game was ~70 per 36
                'reb_per36': 40,  # Wilt's 55-reb game was ~40 per 36
                'ast_per36': 30,  # Scott Skiles' 30-ast game was ~35 per 36
                'stl_per36': 15,  # Historic high was ~11 per 36
                'blk_per36': 15,  # Historic high was ~10 per 36
                'TOV_per36': 20,  # Reasonable cap for turnovers
                'TOV_x_per36': 20,
                'TOV_y_per36': 20
            }
            
            if f'{stat}_per36' in caps:
                cap_value = caps[f'{stat}_per36']
                outliers = (df[f'{stat}_per36'] > cap_value).sum()
                if outliers > 0:
                    logging.info(f"Capping {outliers} outlier values in {stat}_per36 at {cap_value}")
                    df[f'{stat}_per36'] = df[f'{stat}_per36'].clip(upper=cap_value)
            
            # Fill remaining NaNs with 0
            df[f'{stat}_per36'] = df[f'{stat}_per36'].fillna(0)
    
    # Calculate a weight factor for each game based on minutes played
    # Games with more minutes get higher weight in analysis
    max_mins = df['mins'].max()
    df['minutes_weight'] = (df['mins'] / max_mins).clip(lower=0.1, upper=1.0)
    
    logging.info(f"Added low minutes indicators and per-36 stats with appropriate weights")
    
    return df

def check_derived_stats(df, copy=True):
    """
    Check derived and calculated statistics for anomalies
    
    Args:
        df (pd.DataFrame): DataFrame to check
        copy (bool): Whether to make a copy of the dataframe
        
    Returns:
        pd.DataFrame: DataFrame with fixed derived stats
    """
    if copy:
        df = df.copy()
    
    # Try to use the more comprehensive derived checks module
    try:
        try:
            from .data_quality_check_derived import run_derived_checks
            return run_derived_checks(df, copy=False)
        except (ImportError, UnicodeDecodeError):
            # If there's an import error or the file has encoding issues, use the local implementation
            logging.warning("Could not import data_quality_check_derived, using local implementation")
    except Exception as e:
        logging.warning(f"Could not import derived checks module: {str(e)}, using basic checks instead")
    
    # Basic in-place checks 
    
    # Check percentage values
    pct_cols = [col for col in df.columns if any(s in col for s in ['fgp', 'tptfgp', 'ftp', 'FG%', '3P%', 'FT%', 'eFG%', 'TS%'])]
    for col in pct_cols:
        # Make sure percentages are between 0 and 1
        if df[col].max() > 1:
            df[col] = df[col] / 100  # Convert from 0-100 to 0-1 scale
        
        # Clip to valid range
        df[col] = df[col].clip(0, 1)
    
    # Check per-36 minute values
    per36_cols = [col for col in df.columns if col.endswith('_per36')]
    for col in per36_cols:
        # Reasonable maximum values for per-36 stats
        max_val = 60  # Default
        if 'pts' in col:
            max_val = 60
        elif 'reb' in col:
            max_val = 40
        elif 'ast' in col:
            max_val = 30
        elif 'stl' in col or 'blk' in col:
            max_val = 15
        elif 'TOV' in col:
            max_val = 20
        
        # Clip to valid range
        df[col] = df[col].clip(0, max_val)
    
    # Return the fixed dataframe
    return df

def run_all_quality_checks(df, min_minutes=10, check_derived=True, copy=True):
    """
    Run all data quality checks
    
    Args:
        df (pd.DataFrame): DataFrame to process
        min_minutes (int): Minimum minutes threshold for low-minute players
        check_derived (bool): Whether to check derived statistics
        copy (bool): Whether to make a copy of the dataframe
    
    Returns:
        pd.DataFrame: DataFrame with all quality checks applied
    """
    if copy:
        df = df.copy()
    
    # Log the initial state
    logging.info(f"Running data quality checks on DataFrame with {df.shape[0]} rows and {df.shape[1]} columns")
    
    # Run each quality check in sequence
    df = fix_invalid_values(df, copy=False)
    df = fix_na_values(df, copy=False)
    df = resolve_turnover_columns(df, copy=False)
    df = handle_low_minute_players(df, min_minutes=min_minutes, copy=False)
    
    # Run checks on derived statistics if requested
    if check_derived:
        df = check_derived_stats(df, copy=False)
    
    # Log the final state
    logging.info(f"Completed data quality checks, DataFrame now has {df.shape[0]} rows and {df.shape[1]} columns")
    
    return df

if __name__ == "__main__":
    import os
    import argparse
    from datetime import datetime
    
    # Define command line arguments
    parser = argparse.ArgumentParser(description="Run data quality checks on NBA data")
    
    parser.add_argument("--input-file", type=str, help="Path to input CSV file")
    parser.add_argument("--output-file", type=str, help="Path to save cleaned CSV file")
    parser.add_argument("--min-minutes", type=int, default=10, help="Minimum minutes threshold for low-minute players")
    parser.add_argument("--skip-derived-checks", action="store_true", help="Skip checks on derived statistics")
    
    args = parser.parse_args()
    
    # Check if input file specified
    if not args.input_file:
        # Try to find the latest processed data file
        try:
            from config import get_processed_data_path
            input_path = get_processed_data_path()
        except ImportError:
            # Default fallback
            current_date = datetime.now().strftime("%Y%m%d")
            input_path = f"/Users/lukesmac/Projects/nbaModel/data/processed/processed_nba_data_{current_date}.csv"
        
        # If today's file doesn't exist, try to find the latest one
        if not os.path.exists(input_path):
            processed_dir = os.path.dirname(input_path)
            if os.path.exists(processed_dir):
                processed_files = sorted(
                    [f for f in os.listdir(processed_dir) if f.startswith("processed_nba_data_")],
                    reverse=True
                )
                if processed_files:
                    input_path = os.path.join(processed_dir, processed_files[0])
                else:
                    logging.error("No processed data file found")
                    exit(1)
    else:
        input_path = args.input_file
    
    # Load the data
    try:
        logging.info(f"Loading data from: {input_path}")
        df = pd.read_csv(input_path)
        logging.info(f"Loaded data with {df.shape[0]} rows and {df.shape[1]} columns")
    except Exception as e:
        logging.error(f"Error loading data: {str(e)}")
        exit(1)
    
    # Run quality checks
    cleaned_df = run_all_quality_checks(
        df, 
        min_minutes=args.min_minutes,
        check_derived=not args.skip_derived_checks
    )
    
    # Determine output path
    if args.output_file:
        output_path = args.output_file
    else:
        # Use the input filename with "_cleaned" suffix
        base_path = os.path.splitext(input_path)[0]
        output_path = f"{base_path}_cleaned.csv"
    
    # Save the cleaned data
    cleaned_df.to_csv(output_path, index=False)
    logging.info(f"Saved cleaned data to {output_path}")
    
    # Print summary of changes
    na_before = df.isna().sum().sum()
    na_after = cleaned_df.isna().sum().sum()
    
    print("\nData Quality Check Summary:")
    print(f"Rows: {df.shape[0]} (unchanged)")
    print(f"Columns: {df.shape[1]} → {cleaned_df.shape[1]} ({cleaned_df.shape[1] - df.shape[1]} added)")
    print(f"Missing values: {na_before} → {na_after} ({na_before - na_after} resolved)")
    print(f"Low-minute players: {cleaned_df['predominantly_low_minutes_player'].sum()}")
    
    # Check for TOV resolution
    if ('TOV_x' in df.columns or 'TOV_y' in df.columns) and 'TOV' in cleaned_df.columns:
        print("✓ Resolved turnover column inconsistencies")
    
    print("\nData quality checks complete!")