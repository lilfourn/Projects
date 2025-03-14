import os
import glob
import re
import shutil
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Base data directory
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

# Get current date for file naming
CURRENT_DATE = datetime.now().strftime("%Y%m%d")

# Data subdirectories
PLAYER_STATS_DIR = os.path.join(DATA_DIR, "player_stats")
PLAYER_GAME_STATS_DIR = os.path.join(DATA_DIR, "playerGameStats")
PLAYER_INFO_DIR = os.path.join(DATA_DIR, "playerInfo")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
ENGINEERED_DATA_DIR = os.path.join(DATA_DIR, "engineered")
STANDINGS_DIR = os.path.join(DATA_DIR, "standings")
SCHEDULES_DIR = os.path.join(DATA_DIR, "schedules")
PROJECTIONS_DIR = os.path.join(DATA_DIR, "projections")
MODELS_DIR = os.path.join(os.path.dirname(DATA_DIR), "models")

# Data retention configuration
DEFAULT_RETENTION_DAYS = 30  # Default number of days to keep files
DEFAULT_KEEP_LATEST = 3      # Default number of latest files to always keep

# File paths (with date formatting)
def get_player_averages_path(date=None):
    """Get the path to player averages file for a specific date"""
    date_str = date or CURRENT_DATE
    return os.path.join(PLAYER_STATS_DIR, f"player_averages_{date_str}.csv")

def get_player_game_stats_path(date=None, season="2025"):
    """Get the path to player game stats file for a specific date"""
    date_str = date or CURRENT_DATE
    return os.path.join(PLAYER_GAME_STATS_DIR, f"all_player_games_{season}_{date_str}.csv")

def get_player_info_path(date=None):
    """Get the path to player info file for a specific date"""
    date_str = date or CURRENT_DATE
    return os.path.join(PLAYER_INFO_DIR, f"player_info_{date_str}.csv")

def get_processed_data_path(date=None):
    """Get the path to processed data file for a specific date"""
    date_str = date or CURRENT_DATE
    return os.path.join(PROCESSED_DATA_DIR, f"processed_nba_data_{date_str}.csv")

def get_engineered_data_path(date=None):
    """Get the path to engineered data file for a specific date"""
    date_str = date or CURRENT_DATE
    return os.path.join(ENGINEERED_DATA_DIR, f"engineered_nba_data_{date_str}.csv")

def get_team_ratings_path(date=None):
    """Get the path to team ratings file for a specific date"""
    date_str = date or CURRENT_DATE
    return os.path.join(STANDINGS_DIR, f"team_ratings_{date_str}.csv")

def get_model_path(model_type="dt", date=None):
    """Get the path to model file for a specific date"""
    date_str = date or CURRENT_DATE
    return os.path.join(MODELS_DIR, f"nba_{model_type}_model_{date_str}.joblib")

def get_metrics_path(model_type="dt", date=None):
    """Get the path to metrics file for a specific date"""
    date_str = date or CURRENT_DATE
    return os.path.join(MODELS_DIR, f"nba_{model_type}_metrics_{date_str}.json")

# Memory-efficient loading parameters
CHUNK_SIZE = 10000  # Number of rows to load at a time when processing in chunks
MINIMUM_MEMORY_AVAILABLE = 0.2  # Minimum available memory (as a fraction of total) before using chunking

# Essential columns for different data types
SEASON_AVG_ESSENTIAL_COLUMNS = [
    'Player', 'Tm', 'Season_Year', 'MP', 'Age', 'PTS', 'TRB', 'AST', 'STL', 'BLK', 
    'TOV', 'FG%', '3P%', 'FT%', 'USG%', 'WS/48', 'OBPM', 'DBPM', 'BPM'
]

GAME_STATS_ESSENTIAL_COLUMNS = [
    'playerID', 'longName', 'gameID', 'teamID', 'teamAbv', 'pts', 'reb', 'ast', 
    'stl', 'blk', 'TOV', 'fgm', 'fga', 'tptfgm', 'tptfga', 'ftm', 'fta', 
    'OffReb', 'DefReb', 'PF', 'plusMinus', 'mins'
]

# Create directories if they don't exist
def ensure_directories_exist():
    """Ensure all data directories exist"""
    directories = [
        DATA_DIR, PLAYER_STATS_DIR, PLAYER_GAME_STATS_DIR, PLAYER_INFO_DIR,
        PROCESSED_DATA_DIR, ENGINEERED_DATA_DIR, STANDINGS_DIR, SCHEDULES_DIR,
        PROJECTIONS_DIR, MODELS_DIR
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def extract_date_from_filename(filename):
    """
    Extract date from filename using regex pattern
    
    Args:
        filename (str): Filename to extract date from
        
    Returns:
        str or None: Date string if found, None otherwise
    """
    # Pattern to match dates in YYYYMMDD format
    pattern = r'_(\d{8})\.'
    match = re.search(pattern, filename)
    if match:
        return match.group(1)
    return None

def cleanup_old_data(directory, pattern, max_age_days=DEFAULT_RETENTION_DAYS, keep_latest=DEFAULT_KEEP_LATEST, dry_run=False):
    """
    Clean up old data files, keeping only the latest N files and files newer than max_age_days
    
    Args:
        directory (str): Directory containing files to clean up
        pattern (str): Glob pattern to match files
        max_age_days (int): Maximum age in days to keep files
        keep_latest (int): Minimum number of latest files to keep
        dry_run (bool): If True, only print what would be deleted without deleting
        
    Returns:
        int: Number of files deleted
    """
    if not os.path.exists(directory):
        logging.warning(f"Directory does not exist: {directory}")
        return 0
    
    # Get all matching files
    full_pattern = os.path.join(directory, pattern)
    files = glob.glob(full_pattern)
    
    if not files:
        logging.info(f"No files matching pattern {pattern} found in {directory}")
        return 0
    
    # Get file info with dates
    file_info = []
    for file_path in files:
        filename = os.path.basename(file_path)
        date_str = extract_date_from_filename(filename)
        
        if date_str:
            try:
                # Convert date string to datetime object
                file_date = datetime.strptime(date_str, '%Y%m%d')
                file_info.append((file_path, file_date, filename))
            except ValueError:
                # If date conversion fails, use file modification time
                mtime = os.path.getmtime(file_path)
                file_date = datetime.fromtimestamp(mtime)
                file_info.append((file_path, file_date, filename))
        else:
            # If no date in filename, use file modification time
            mtime = os.path.getmtime(file_path)
            file_date = datetime.fromtimestamp(mtime)
            file_info.append((file_path, file_date, filename))
    
    # Sort files by date (newest first)
    file_info.sort(key=lambda x: x[1], reverse=True)
    
    # Always keep the latest N files
    files_to_keep = file_info[:keep_latest]
    files_to_check = file_info[keep_latest:]
    
    # Calculate cutoff date
    cutoff_date = datetime.now() - timedelta(days=max_age_days)
    
    # Determine which files to delete (older than cutoff_date)
    files_to_delete = [f for f in files_to_check if f[1] < cutoff_date]
    
    # Delete files or print what would be deleted
    deleted_count = 0
    for file_path, file_date, filename in files_to_delete:
        if dry_run:
            logging.info(f"Would delete: {filename} (date: {file_date.strftime('%Y-%m-%d')})")
        else:
            try:
                os.remove(file_path)
                logging.info(f"Deleted: {filename} (date: {file_date.strftime('%Y-%m-%d')})")
                deleted_count += 1
            except Exception as e:
                logging.error(f"Error deleting {filename}: {str(e)}")
    
    # Log summary
    if dry_run:
        logging.info(f"Would delete {len(files_to_delete)} files from {directory}")
    else:
        logging.info(f"Deleted {deleted_count} files from {directory}")
    
    return deleted_count

def cleanup_all_data_directories(max_age_days=DEFAULT_RETENTION_DAYS, keep_latest=DEFAULT_KEEP_LATEST, dry_run=False):
    """
    Clean up old data files from all data directories
    
    Args:
        max_age_days (int): Maximum age in days to keep files
        keep_latest (int): Minimum number of latest files to keep
        dry_run (bool): If True, only print what would be deleted without deleting
        
    Returns:
        int: Total number of files deleted
    """
    # Ensure directories exist
    ensure_directories_exist()
    
    total_deleted = 0
    
    # Define cleanup tasks (directory, pattern)
    cleanup_tasks = [
        (PLAYER_STATS_DIR, "player_averages_*.csv"),
        (PLAYER_GAME_STATS_DIR, "all_player_games_*.csv"),
        (PLAYER_INFO_DIR, "player_info_*.csv"),
        (PROCESSED_DATA_DIR, "processed_nba_data_*.csv"),
        (ENGINEERED_DATA_DIR, "engineered_nba_data_*.csv"),
        (STANDINGS_DIR, "team_ratings_*.csv"),
        (MODELS_DIR, "nba_*_model_*.joblib"),
        (MODELS_DIR, "nba_*_metrics_*.json"),
        (MODELS_DIR, "feature_importance_*.json")
    ]
    
    # Run cleanup for each task
    for directory, pattern in cleanup_tasks:
        deleted = cleanup_old_data(
            directory=directory,
            pattern=pattern,
            max_age_days=max_age_days,
            keep_latest=keep_latest,
            dry_run=dry_run
        )
        total_deleted += deleted
    
    # Log summary
    if dry_run:
        logging.info(f"Would delete {total_deleted} files in total")
    else:
        logging.info(f"Deleted {total_deleted} files in total")
    
    return total_deleted