import pandas as pd
import numpy as np
import os
import psutil
from datetime import datetime
import logging
import glob
try:
    from src.config import (
        CURRENT_DATE, SEASON_AVG_ESSENTIAL_COLUMNS, GAME_STATS_ESSENTIAL_COLUMNS,
        CHUNK_SIZE, MINIMUM_MEMORY_AVAILABLE, ensure_directories_exist,
        get_player_averages_path, get_player_game_stats_path, get_processed_data_path
    )
except ImportError:
    # Try importing without the src prefix
    from config import (
        CURRENT_DATE, SEASON_AVG_ESSENTIAL_COLUMNS, GAME_STATS_ESSENTIAL_COLUMNS,
        CHUNK_SIZE, MINIMUM_MEMORY_AVAILABLE, ensure_directories_exist,
        get_player_averages_path, get_player_game_stats_path, get_processed_data_path
    )

# Import data quality module if available
try:
    from src.data_quality import run_all_quality_checks
    QUALITY_CHECKS_AVAILABLE = True
except ImportError:
    QUALITY_CHECKS_AVAILABLE = False
    logging.warning("Data quality module not available. Install with 'pip install -e .' for data quality checks.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def get_available_memory_percentage():
    """
    Get available memory as a percentage of total memory
    
    Returns:
        float: Percentage of available memory (0.0 to 1.0)
    """
    memory = psutil.virtual_memory()
    return memory.available / memory.total

def find_latest_file(base_path, pattern):
    """
    Find the latest file matching a pattern
    
    Args:
        base_path (str): Base directory to search in
        pattern (str): File pattern to match
        
    Returns:
        str: Path to the latest file, or None if no files found
    """
    try:
        if not os.path.exists(base_path):
            logging.warning(f"Directory not found: {base_path}")
            return None
            
        # Find all matching files
        matching_files = sorted(
            glob.glob(os.path.join(base_path, pattern)),
            reverse=True
        )
        
        if matching_files:
            return matching_files[0]
        else:
            logging.warning(f"No files found matching {pattern} in {base_path}")
            return None
    except Exception as e:
        logging.error(f"Error finding latest file: {str(e)}")
        return None

def load_season_averages(file_path, columns=None):
    """
    Load season averages data from CSV file with lazy loading
    
    Args:
        file_path (str): Path to the season averages CSV file
        columns (list, optional): List of specific columns to load
        
    Returns:
        pd.DataFrame: DataFrame containing season averages
    """
    try:
        # Check if we should use column filtering (lazy loading)
        if columns is None:
            columns = SEASON_AVG_ESSENTIAL_COLUMNS
        
        # Check available memory to determine if chunking is needed
        if get_available_memory_percentage() < MINIMUM_MEMORY_AVAILABLE:
            logging.info("Low memory available. Loading in chunks...")
            
            # Read file in chunks and select only required columns
            chunk_list = []
            for chunk in pd.read_csv(file_path, usecols=columns, chunksize=CHUNK_SIZE):
                chunk_list.append(chunk)
            
            # Combine chunks
            df = pd.concat(chunk_list)
            logging.info(f"Loaded season averages data with {df.shape[0]} rows and {df.shape[1]} columns (chunked)")
        else:
            # Read the entire file at once with column filtering
            df = pd.read_csv(file_path, usecols=columns)
            logging.info(f"Loaded season averages data with {df.shape[0]} rows and {df.shape[1]} columns")
        
        return df
    
    except Exception as e:
        logging.error(f"Error loading season averages data: {str(e)}")
        return None

def load_game_stats(file_path, columns=None):
    """
    Load game-by-game stats from CSV file with lazy loading
    
    Args:
        file_path (str): Path to the game stats CSV file
        columns (list, optional): List of specific columns to load
        
    Returns:
        pd.DataFrame: DataFrame containing game-by-game stats
    """
    try:
        # Check if we should use column filtering (lazy loading)
        if columns is None:
            columns = GAME_STATS_ESSENTIAL_COLUMNS
        
        # Check available memory to determine if chunking is needed
        if get_available_memory_percentage() < MINIMUM_MEMORY_AVAILABLE:
            logging.info("Low memory available. Loading in chunks...")
            
            # Read file in chunks and select only required columns
            chunk_list = []
            for chunk in pd.read_csv(file_path, usecols=columns, chunksize=CHUNK_SIZE):
                chunk_list.append(chunk)
            
            # Combine chunks
            df = pd.concat(chunk_list)
            logging.info(f"Loaded game stats data with {df.shape[0]} rows and {df.shape[1]} columns (chunked)")
        else:
            # Read the entire file at once with column filtering
            df = pd.read_csv(file_path, usecols=columns)
            logging.info(f"Loaded game stats data with {df.shape[0]} rows and {df.shape[1]} columns")
        
        return df
    
    except Exception as e:
        logging.error(f"Error loading game stats data: {str(e)}")
        return None

def clean_season_averages(df):
    """
    Clean and preprocess season averages data
    
    Args:
        df (pd.DataFrame): DataFrame containing season averages
        
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    # Make a copy to avoid modifying the original
    df_clean = df.copy()
    
    # Drop rows with missing key values
    df_clean = df_clean.dropna(subset=['Player', 'Tm', 'Season_Year'])
    
    # Convert numeric columns to float
    numeric_cols = df_clean.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    # Handle missing values in numeric columns
    df_clean[numeric_cols] = df_clean[numeric_cols].fillna(0)
    
    # Create a unique player identifier
    df_clean['player_id'] = df_clean['Player'] + '_' + df_clean['Tm'] + '_' + df_clean['Season_Year'].astype(str)
    
    logging.info(f"Cleaned season averages data with {df_clean.shape[0]} rows remaining")
    return df_clean

def clean_game_stats(df):
    """
    Clean and preprocess game-by-game stats
    
    Args:
        df (pd.DataFrame): DataFrame containing game-by-game stats
        
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    # Make a copy to avoid modifying the original
    df_clean = df.copy()
    
    # Extract date from gameID (format: YYYYMMDD_TEAM@TEAM)
    df_clean['game_date'] = df_clean['gameID'].str.split('_', expand=True)[0]
    df_clean['game_date'] = pd.to_datetime(df_clean['game_date'], format='%Y%m%d')
    
    # Convert string numeric values to float
    numeric_cols = ['pts', 'reb', 'ast', 'stl', 'blk', 'TOV', 'fgm', 'fga', 'tptfgm', 
                   'tptfga', 'ftm', 'fta', 'OffReb', 'DefReb', 'PF', 'plusMinus', 'mins']
    
    for col in numeric_cols:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    # Calculate percentages if not already present
    if 'fgp' not in df_clean.columns and 'fgm' in df_clean.columns and 'fga' in df_clean.columns:
        df_clean['fgp'] = (df_clean['fgm'] / df_clean['fga'].replace(0, np.nan)) * 100
    
    if 'tptfgp' not in df_clean.columns and 'tptfgm' in df_clean.columns and 'tptfga' in df_clean.columns:
        df_clean['tptfgp'] = (df_clean['tptfgm'] / df_clean['tptfga'].replace(0, np.nan)) * 100
    
    if 'ftp' not in df_clean.columns and 'ftm' in df_clean.columns and 'fta' in df_clean.columns:
        df_clean['ftp'] = (df_clean['ftm'] / df_clean['fta'].replace(0, np.nan)) * 100
    
    # Handle missing values
    df_clean = df_clean.fillna(0)
    
    # Create a unique player identifier that can match with season averages
    if 'playerID' in df_clean.columns and 'teamID' in df_clean.columns:
        df_clean['player_team_id'] = df_clean['playerID'].astype(str) + '_' + df_clean['teamID'].astype(str)
    
    logging.info(f"Cleaned game stats data with {df_clean.shape[0]} rows")
    return df_clean

def create_season_to_date_stats(game_stats_df):
    """
    Create season-to-date averages for each player before each game
    
    Args:
        game_stats_df (pd.DataFrame): DataFrame containing game-by-game stats
        
    Returns:
        pd.DataFrame: DataFrame with season-to-date stats for each game
    """
    # Sort by player and date
    game_stats_df = game_stats_df.sort_values(['playerID', 'game_date'])
    
    # Get the list of stats to compute running averages for
    stat_cols = ['pts', 'reb', 'ast', 'stl', 'blk', 'TOV', 'fgm', 'fga', 'fgp',
                'tptfgm', 'tptfga', 'tptfgp', 'ftm', 'fta', 'ftp', 
                'OffReb', 'DefReb', 'PF', 'plusMinus', 'mins']
    
    # Only include columns that actually exist in the dataframe
    stat_cols = [col for col in stat_cols if col in game_stats_df.columns]
    
    # Create empty dataframe to store results
    result_df = pd.DataFrame()
    
    # Group by player
    for player_id, player_games in game_stats_df.groupby('playerID'):
        # Calculate expanding mean (season-to-date average)
        for stat in stat_cols:
            player_games[f'std_{stat}'] = player_games[stat].expanding().mean().shift(1)
        
        # Calculate expanding standard deviation (measure of consistency)
        for stat in stat_cols:
            player_games[f'std_{stat}_std'] = player_games[stat].expanding().std().shift(1)
        
        # Calculate form metrics (last 5 games average)
        for stat in stat_cols:
            player_games[f'last5_{stat}'] = player_games[stat].rolling(window=5).mean().shift(1)
        
        # Calculate number of games played so far
        player_games['games_played'] = range(len(player_games))
        
        # Add to result
        result_df = pd.concat([result_df, player_games])
    
    # Replace NaNs with 0s for the first games where no prior data exists
    std_cols = [col for col in result_df.columns if col.startswith('std_') or col.startswith('last5_')]
    result_df[std_cols] = result_df[std_cols].fillna(0)
    
    logging.info(f"Created season-to-date stats with {result_df.shape[0]} rows and {result_df.shape[1]} columns")
    return result_df

def merge_season_and_game_data(season_avg_df, game_stats_df):
    """
    Merge season averages with game-by-game stats
    
    Args:
        season_avg_df (pd.DataFrame): DataFrame containing season averages
        game_stats_df (pd.DataFrame): DataFrame containing game-by-game stats with season-to-date features
        
    Returns:
        pd.DataFrame: Merged DataFrame
    """
    # Create a mapping between playerID in game stats and Player in season averages
    # This would typically use a player mapping table, but for simplicity we'll try direct matching
    
    # Extract the season year from the game date
    game_stats_df['season_year'] = game_stats_df['game_date'].dt.year
    # Adjust for NBA season (Oct-Jun spans two years)
    game_stats_df.loc[game_stats_df['game_date'].dt.month >= 10, 'season_year'] += 1
    
    # Prepare for merge
    season_avg_df_copy = season_avg_df.copy()
    game_stats_df_copy = game_stats_df.copy()
    
    # Convert playerID to string if it's not already
    if 'playerID' in game_stats_df_copy.columns:
        game_stats_df_copy['playerID'] = game_stats_df_copy['playerID'].astype(str)
    
    # Try to merge based on player name and season
    # This is a simplified approach - in practice, you would need a more robust mapping
    merged_df = pd.merge(
        game_stats_df_copy,
        season_avg_df_copy,
        left_on=['longName', 'season_year'],
        right_on=['Player', 'Season_Year'],
        how='left'
    )
    
    # Check merge success
    merge_success_count = merged_df['Player'].notna().sum()
    logging.info(f"Successfully merged {merge_success_count} out of {len(game_stats_df)} game records with season averages")
    
    return merged_df

def save_processed_data(df, output_path):
    """
    Save processed data to CSV file
    
    Args:
        df (pd.DataFrame): DataFrame to save
        output_path (str): Path to save the CSV file
    """
    try:
        df.to_csv(output_path, index=False)
        logging.info(f"Saved processed data to {output_path}")
    except Exception as e:
        logging.error(f"Error saving processed data: {str(e)}")

def create_season_to_date_stats_incremental(game_stats_df, increment_size=1000):
    """
    Create season-to-date averages for each player before each game using incremental processing
    
    Args:
        game_stats_df (pd.DataFrame): DataFrame containing game-by-game stats
        increment_size (int): Number of rows to process at once
        
    Returns:
        pd.DataFrame: DataFrame with season-to-date stats for each game
    """
    # Sort by player and date
    game_stats_df = game_stats_df.sort_values(['playerID', 'game_date'])
    
    # Get the list of stats to compute running averages for
    stat_cols = ['pts', 'reb', 'ast', 'stl', 'blk', 'TOV', 'fgm', 'fga', 'fgp',
                'tptfgm', 'tptfga', 'tptfgp', 'ftm', 'fta', 'ftp', 
                'OffReb', 'DefReb', 'PF', 'plusMinus', 'mins']
    
    # Only include columns that actually exist in the dataframe
    stat_cols = [col for col in stat_cols if col in game_stats_df.columns]
    
    # Get unique player IDs
    player_ids = game_stats_df['playerID'].unique()
    
    # Create empty dataframe to store results
    result_df = pd.DataFrame()
    
    # Process players in batches to save memory
    batch_size = min(50, len(player_ids))  # Process up to 50 players at a time
    player_batches = [player_ids[i:i + batch_size] for i in range(0, len(player_ids), batch_size)]
    
    for batch_idx, player_batch in enumerate(player_batches):
        logging.info(f"Processing player batch {batch_idx + 1}/{len(player_batches)} ({len(player_batch)} players)")
        
        # Filter for current batch of players
        batch_df = game_stats_df[game_stats_df['playerID'].isin(player_batch)].copy()
        
        # Group by player and process each player separately
        for player_id, player_games in batch_df.groupby('playerID'):
            # Calculate expanding mean (season-to-date average)
            for stat in stat_cols:
                player_games[f'std_{stat}'] = player_games[stat].expanding().mean().shift(1)
            
            # Calculate expanding standard deviation (measure of consistency)
            for stat in stat_cols:
                player_games[f'std_{stat}_std'] = player_games[stat].expanding().std().shift(1)
            
            # Calculate form metrics (last 5 games average)
            for stat in stat_cols:
                player_games[f'last5_{stat}'] = player_games[stat].rolling(window=5).mean().shift(1)
            
            # Calculate number of games played so far
            player_games['games_played'] = range(len(player_games))
            
            # Add to result
            result_df = pd.concat([result_df, player_games])
        
        # Clear batch data to free memory
        del batch_df
    
    # Replace NaNs with 0s for the first games where no prior data exists
    std_cols = [col for col in result_df.columns if col.startswith('std_') or col.startswith('last5_')]
    result_df[std_cols] = result_df[std_cols].fillna(0)
    
    logging.info(f"Created season-to-date stats with {result_df.shape[0]} rows and {result_df.shape[1]} columns")
    return result_df

def save_processed_data_in_chunks(df, output_path, chunk_size=10000):
    """
    Save processed data to CSV file in chunks to reduce memory usage
    
    Args:
        df (pd.DataFrame): DataFrame to save
        output_path (str): Path to save the CSV file
        chunk_size (int): Number of rows to write at a time
    """
    try:
        # Get the total number of chunks
        total_chunks = (len(df) + chunk_size - 1) // chunk_size
        
        # Write the first chunk with headers
        first_chunk = df.iloc[:min(chunk_size, len(df))]
        first_chunk.to_csv(output_path, index=False, mode='w')
        
        # Write the remaining chunks without headers
        if total_chunks > 1:
            for i in range(1, total_chunks):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, len(df))
                
                chunk = df.iloc[start_idx:end_idx]
                chunk.to_csv(output_path, index=False, mode='a', header=False)
                
                # Log progress
                if i % 5 == 0 or i == total_chunks - 1:
                    logging.info(f"Saved chunk {i + 1}/{total_chunks}")
        
        logging.info(f"Saved processed data to {output_path}")
    except Exception as e:
        logging.error(f"Error saving processed data: {str(e)}")

def main(season_avg_path=None, game_stats_path=None, output_dir=None, low_memory_mode=False, 
         run_quality_checks=True, min_minutes_threshold=10):
    """
    Main function to process data with memory-efficient options
    
    Args:
        season_avg_path (str): Path to season averages CSV
        game_stats_path (str): Path to game stats CSV
        output_dir (str): Directory to save processed data
        low_memory_mode (bool): Whether to use more aggressive memory saving techniques
        run_quality_checks (bool): Whether to run additional data quality checks
        min_minutes_threshold (int): Minimum minutes threshold for low-minute players
        
    Returns:
        pd.DataFrame or None: Processed data if successful, None otherwise
    """
    # Ensure all directories exist
    ensure_directories_exist()
    
    # Set default paths using config utilities if not provided
    if season_avg_path is None:
        # Try today's file first
        season_avg_path = get_player_averages_path()
        
        # If today's file doesn't exist, try to find the latest one
        if not os.path.exists(season_avg_path):
            latest_file = find_latest_file(os.path.dirname(season_avg_path), "player_averages_*.csv")
            if latest_file:
                season_avg_path = latest_file
            else:
                logging.error("No player averages file found")
                return None
    
    if game_stats_path is None:
        # Try today's file first
        game_stats_path = get_player_game_stats_path()
        
        # If today's file doesn't exist, try to find the latest one
        if not os.path.exists(game_stats_path):
            latest_file = find_latest_file(os.path.dirname(game_stats_path), "all_player_games_*.csv")
            if latest_file:
                game_stats_path = latest_file
            else:
                logging.error("No player game stats file found")
                return None
    
    if output_dir is None:
        output_path = get_processed_data_path()
    else:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"processed_nba_data_{CURRENT_DATE}.csv")
    
    # Load data with lazy loading (only essential columns)
    logging.info(f"Loading season averages from: {season_avg_path}")
    season_avg_df = load_season_averages(season_avg_path)
    
    logging.info(f"Loading game stats from: {game_stats_path}")
    game_stats_df = load_game_stats(game_stats_path)
    
    if season_avg_df is None or game_stats_df is None:
        logging.error("Failed to load one or more required datasets")
        return None
    
    # Clean data
    season_avg_clean = clean_season_averages(season_avg_df)
    # Clear original dataframe to save memory
    del season_avg_df
    
    game_stats_clean = clean_game_stats(game_stats_df)
    # Clear original dataframe to save memory
    del game_stats_df
    
    # Create season-to-date stats (using incremental processing in low memory mode)
    if low_memory_mode:
        game_stats_with_std = create_season_to_date_stats_incremental(game_stats_clean)
    else:
        game_stats_with_std = create_season_to_date_stats(game_stats_clean)
    
    # Clear cleaned game stats to save memory
    del game_stats_clean
    
    # Merge data
    merged_data = merge_season_and_game_data(season_avg_clean, game_stats_with_std)
    
    # Clear intermediate dataframes to save memory
    del season_avg_clean
    del game_stats_with_std
    
    # Run additional data quality checks if enabled and available
    if run_quality_checks and QUALITY_CHECKS_AVAILABLE:
        logging.info("Running additional data quality checks...")
        merged_data = run_all_quality_checks(merged_data, min_minutes=min_minutes_threshold)
        logging.info("Data quality checks completed")
    elif run_quality_checks and not QUALITY_CHECKS_AVAILABLE:
        logging.warning("Data quality checks requested but not available")
    
    # Save processed data (in chunks if in low memory mode)
    if low_memory_mode:
        save_processed_data_in_chunks(merged_data, output_path)
    else:
        save_processed_data(merged_data, output_path)
    
    return merged_data

if __name__ == "__main__":
    import argparse
    
    # Check if psutil is available
    try:
        import psutil
    except ImportError:
        print("Warning: psutil module not found. Install with 'pip install psutil' for memory-efficient loading.")
    
    # Import cleanup functions from config
    try:
        from src.config import cleanup_all_data_directories, DEFAULT_RETENTION_DAYS, DEFAULT_KEEP_LATEST
    except ImportError:
        # Try importing without the src prefix
        from config import cleanup_all_data_directories, DEFAULT_RETENTION_DAYS, DEFAULT_KEEP_LATEST
    
    parser = argparse.ArgumentParser(description='Process NBA data for modeling')
    parser.add_argument('--season-avg-path', type=str, help='Path to season averages CSV')
    parser.add_argument('--game-stats-path', type=str, help='Path to game stats CSV')
    parser.add_argument('--output-dir', type=str, help='Directory to save processed data')
    parser.add_argument('--low-memory', action='store_true', help='Use more aggressive memory-saving techniques')
    parser.add_argument('--columns', type=str, nargs='+', help='Specific columns to load (comma-separated for each file type)')
    
    # Data quality options
    quality_group = parser.add_argument_group('Data Quality Options')
    quality_group.add_argument('--skip-quality-checks', action='store_true', 
                            help='Skip additional data quality checks')
    quality_group.add_argument('--min-minutes', type=int, default=10, 
                            help='Minimum minutes threshold for low-minute players (default: 10)')
    
    # Data cleanup options
    cleanup_group = parser.add_argument_group('Data Cleanup Options')
    cleanup_group.add_argument('--cleanup', action='store_true', help='Clean up old data files after processing')
    cleanup_group.add_argument('--cleanup-dry-run', action='store_true', help='Show what would be deleted without actually deleting')
    cleanup_group.add_argument('--max-age', type=int, default=DEFAULT_RETENTION_DAYS, 
                            help=f'Maximum age in days to keep files (default: {DEFAULT_RETENTION_DAYS})')
    cleanup_group.add_argument('--keep-latest', type=int, default=DEFAULT_KEEP_LATEST, 
                            help=f'Minimum number of latest files to keep (default: {DEFAULT_KEEP_LATEST})')
    
    args = parser.parse_args()
    
    # Run the main function with the provided arguments
    result = main(
        season_avg_path=args.season_avg_path,
        game_stats_path=args.game_stats_path,
        output_dir=args.output_dir,
        low_memory_mode=args.low_memory,
        run_quality_checks=not args.skip_quality_checks,
        min_minutes_threshold=args.min_minutes
    )
    
    # Clean up old data files if requested
    if args.cleanup or args.cleanup_dry_run:
        logging.info("Starting data cleanup...")
        
        # Run cleanup
        cleanup_all_data_directories(
            max_age_days=args.max_age,
            keep_latest=args.keep_latest,
            dry_run=args.cleanup_dry_run
        )
        
        logging.info("Data cleanup completed")