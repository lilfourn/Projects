#!/usr/bin/env python3
# Data Cleanup Utility
# This script cleans up old data files to ensure only the latest or most important files are kept

import os
import glob
import logging
import argparse
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Define project root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

# Define directories to clean up
DATA_DIRS = {
    'data_raw': os.path.join(PROJECT_ROOT, 'data', 'raw'),
    'data_processed': os.path.join(PROJECT_ROOT, 'data', 'processed'),
    'data_engineered': os.path.join(PROJECT_ROOT, 'data', 'engineered'),
    'models_pts': os.path.join(PROJECT_ROOT, 'models', 'pts'),
    'models_reb': os.path.join(PROJECT_ROOT, 'models', 'reb'),
    'models_ast': os.path.join(PROJECT_ROOT, 'models', 'ast'),
    'visualizations_scoring': os.path.join(PROJECT_ROOT, 'visualizations', 'scoring'),
    'visualizations_rebounds': os.path.join(PROJECT_ROOT, 'visualizations', 'rebounds'),
    'visualizations_assists': os.path.join(PROJECT_ROOT, 'visualizations', 'assists')
}

# Define file patterns to match in each directory
FILE_PATTERNS = {
    'data_raw': ['*.csv', '*.json'],
    'data_processed': ['*.csv', '*.json'],
    'data_engineered': ['*.csv', '*.json', '*.pkl'],
    'models_pts': ['*.joblib', '*.json', '*.pkl'],
    'models_reb': ['*.joblib', '*.json', '*.pkl'],
    'models_ast': ['*.joblib', '*.json', '*.pkl'],
    'visualizations_scoring': ['*.png', '*.html', '*.json'],
    'visualizations_rebounds': ['*.png', '*.html', '*.json'],
    'visualizations_assists': ['*.png', '*.html', '*.json']
}

def list_files_by_directory():
    """List all files in each data directory"""
    for dir_name, dir_path in DATA_DIRS.items():
        if os.path.exists(dir_path):
            print(f"\n=== Files in {dir_name} directory ===")
            files = os.listdir(dir_path)
            if files:
                for file in sorted(files):
                    file_path = os.path.join(dir_path, file)
                    if os.path.isfile(file_path):
                        mod_time = os.path.getmtime(file_path)
                        mod_date = datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M:%S')
                        size_kb = os.path.getsize(file_path) / 1024
                        print(f"{file} - Modified: {mod_date}, Size: {size_kb:.1f} KB")
            else:
                print("No files found.")
        else:
            print(f"\n=== Directory {dir_name} does not exist ===")

def clean_directory(dir_path, patterns, max_age_days=30, keep_latest=3, dry_run=False):
    """
    Clean up files in a directory based on patterns and age
    
    Args:
        dir_path (str): Directory path to clean
        patterns (list): List of file patterns to match
        max_age_days (int): Maximum age in days to keep files
        keep_latest (int): Minimum number of latest files to keep
        dry_run (bool): Whether to just print what would be deleted without actually deleting
        
    Returns:
        int: Number of files deleted
    """
    if not os.path.exists(dir_path):
        logging.warning(f"Directory does not exist: {dir_path}")
        return 0
    
    # Get current time
    now = datetime.now()
    cutoff_date = now - timedelta(days=max_age_days)
    
    # Keep track of deleted files count
    deleted_count = 0
    
    # Process each pattern
    for pattern in patterns:
        # Find all matching files
        full_pattern = os.path.join(dir_path, pattern)
        matched_files = glob.glob(full_pattern)
        
        if not matched_files:
            logging.info(f"No files matching pattern {pattern} in {dir_path}")
            continue
        
        # Group files by prefix (everything before the date)
        file_groups = {}
        for file_path in matched_files:
            file_name = os.path.basename(file_path)
            
            # Try to find the date part (assuming YYYYMMDD format)
            date_part = None
            for part in file_name.split('_'):
                if len(part) == 8 and part.isdigit():
                    date_part = part
                    break
            
            # Determine prefix (group key)
            if date_part:
                prefix = file_name.replace(date_part, '*')
            else:
                prefix = file_name
            
            if prefix not in file_groups:
                file_groups[prefix] = []
            
            # Get file stats
            mod_time = os.path.getmtime(file_path)
            mod_date = datetime.fromtimestamp(mod_time)
            
            file_groups[prefix].append({
                'path': file_path,
                'name': file_name,
                'mod_time': mod_time,
                'mod_date': mod_date,
                'is_old': mod_date < cutoff_date
            })
        
        # Process each group separately to keep latest N files of each type
        for prefix, files in file_groups.items():
            # Sort files by modification time (newest first)
            sorted_files = sorted(files, key=lambda x: x['mod_time'], reverse=True)
            
            # Keep the latest 'keep_latest' files
            keep_count = min(keep_latest, len(sorted_files))
            files_to_keep = sorted_files[:keep_count]
            
            # Delete older files beyond the keep count
            for file_info in sorted_files[keep_count:]:
                # If it's also older than max_age_days
                if file_info['is_old']:
                    if dry_run:
                        age_days = (now - file_info['mod_date']).days
                        logging.info(f"Would delete: {file_info['name']} (age: {age_days} days)")
                        deleted_count += 1
                    else:
                        try:
                            os.remove(file_info['path'])
                            age_days = (now - file_info['mod_date']).days
                            logging.info(f"Deleted: {file_info['name']} (age: {age_days} days)")
                            deleted_count += 1
                        except Exception as e:
                            logging.error(f"Error deleting {file_info['path']}: {str(e)}")
    
    return deleted_count

def cleanup_all_data_directories(max_age_days=30, keep_latest=3, dry_run=False, force_clean=False):
    """
    Clean up all data directories based on configured patterns
    
    Args:
        max_age_days (int): Maximum age in days to keep files
        keep_latest (int): Minimum number of latest files to keep
        dry_run (bool): Whether to just print what would be deleted without actually deleting
        force_clean (bool): Whether to force cleanup all files (keep only the very latest)
        
    Returns:
        int: Total number of files deleted
    """
    # If force clean is enabled, we keep only the very latest file
    if force_clean:
        keep_latest = 1
        max_age_days = 0  # Delete everything except the latest file
        logging.warning("Force clean mode enabled - only the latest file of each type will be kept!")
    
    total_deleted = 0
    
    for dir_name, dir_path in DATA_DIRS.items():
        patterns = FILE_PATTERNS.get(dir_name, ['*'])
        logging.info(f"Cleaning directory: {dir_name}")
        
        deleted = clean_directory(
            dir_path=dir_path,
            patterns=patterns,
            max_age_days=max_age_days,
            keep_latest=keep_latest,
            dry_run=dry_run
        )
        
        total_deleted += deleted
        
        if dry_run:
            logging.info(f"Would delete {deleted} files from {dir_name}")
        else:
            logging.info(f"Deleted {deleted} files from {dir_name}")
    
    return total_deleted

def clean_all_output_files(dry_run=False):
    """
    Clean all output files (current data files) to start fresh
    
    Args:
        dry_run (bool): Whether to just print what would be deleted without actually deleting
        
    Returns:
        int: Total number of files deleted
    """
    total_deleted = 0
    
    for dir_name, dir_path in DATA_DIRS.items():
        patterns = FILE_PATTERNS.get(dir_name, ['*'])
        logging.info(f"Cleaning all output files in: {dir_name}")
        
        if not os.path.exists(dir_path):
            logging.warning(f"Directory does not exist: {dir_path}")
            continue
        
        # Process each pattern
        for pattern in patterns:
            # Find all matching files
            full_pattern = os.path.join(dir_path, pattern)
            matched_files = glob.glob(full_pattern)
            
            for file_path in matched_files:
                if dry_run:
                    logging.info(f"Would delete: {os.path.basename(file_path)}")
                    total_deleted += 1
                else:
                    try:
                        os.remove(file_path)
                        logging.info(f"Deleted: {os.path.basename(file_path)}")
                        total_deleted += 1
                    except Exception as e:
                        logging.error(f"Error deleting {file_path}: {str(e)}")
    
    return total_deleted

def clean_all_data_models_visualizations(dry_run=False):
    """
    Clean all data, models, and visualizations to start fresh
    
    Args:
        dry_run (bool): Whether to just print what would be deleted without actually deleting
        
    Returns:
        int: Total number of files deleted
    """
    total_deleted = 0
    
    # Define main directories to clean
    main_dirs = {
        'data': os.path.join(PROJECT_ROOT, 'data'),
        'models': os.path.join(PROJECT_ROOT, 'models'),
        'visualizations': os.path.join(PROJECT_ROOT, 'visualizations')
    }
    
    for dir_name, dir_path in main_dirs.items():
        logging.info(f"Cleaning all files in {dir_name} directory")
        
        if not os.path.exists(dir_path):
            logging.warning(f"Directory does not exist: {dir_path}")
            continue
        
        # Get all files in the directory and its subdirectories
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                file_path = os.path.join(root, file)
                
                if dry_run:
                    logging.info(f"Would delete: {file_path}")
                    total_deleted += 1
                else:
                    try:
                        os.remove(file_path)
                        logging.info(f"Deleted: {file_path}")
                        total_deleted += 1
                    except Exception as e:
                        logging.error(f"Error deleting {file_path}: {str(e)}")
        
        # Recreate the directory structure
        if not dry_run:
            # First remove all empty directories
            for root, dirs, files in os.walk(dir_path, topdown=False):
                for dir_name in dirs:
                    try:
                        dir_to_remove = os.path.join(root, dir_name)
                        if not os.listdir(dir_to_remove):  # Check if directory is empty
                            os.rmdir(dir_to_remove)
                            logging.info(f"Removed empty directory: {dir_to_remove}")
                    except Exception as e:
                        logging.error(f"Error removing directory {dir_to_remove}: {str(e)}")
            
            # Recreate standard directory structure
            if dir_name == 'data':
                os.makedirs(os.path.join(dir_path, 'raw'), exist_ok=True)
                os.makedirs(os.path.join(dir_path, 'processed'), exist_ok=True)
                os.makedirs(os.path.join(dir_path, 'engineered'), exist_ok=True)
                logging.info(f"Recreated standard directory structure for {dir_name}")
            elif dir_name == 'models':
                os.makedirs(os.path.join(dir_path, 'pts'), exist_ok=True)
                os.makedirs(os.path.join(dir_path, 'reb'), exist_ok=True)
                os.makedirs(os.path.join(dir_path, 'ast'), exist_ok=True)
                logging.info(f"Recreated standard directory structure for {dir_name}")
            elif dir_name == 'visualizations':
                os.makedirs(os.path.join(dir_path, 'scoring', 'pts'), exist_ok=True)
                os.makedirs(os.path.join(dir_path, 'rebounds', 'reb'), exist_ok=True)
                os.makedirs(os.path.join(dir_path, 'assists', 'ast'), exist_ok=True)
                logging.info(f"Recreated standard directory structure for {dir_name}")
    
    return total_deleted

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data cleanup utility for NBA model')
    
    # Cleanup options
    cleanup_group = parser.add_argument_group('Cleanup Options')
    cleanup_group.add_argument('--list', action='store_true', help='List all files in data directories')
    cleanup_group.add_argument('--clean', action='store_true', help='Clean up old data files')
    cleanup_group.add_argument('--clean-all', action='store_true', help='Clean ALL output files (start fresh)')
    cleanup_group.add_argument('--reset', action='store_true', help='Reset ALL data, models, and visualizations (complete reset)')
    cleanup_group.add_argument('--max-age', type=int, default=30, help='Maximum age in days to keep files (default: 30)')
    cleanup_group.add_argument('--keep-latest', type=int, default=3, help='Number of latest files to keep for each type (default: 3)')
    cleanup_group.add_argument('--dry-run', action='store_true', help="Don't actually delete files, just show what would be deleted")
    cleanup_group.add_argument('--force', action='store_true', help='Force cleanup - keep only the very latest file of each type')
    
    args = parser.parse_args()
    
    # If no action specified, show help
    if not any([args.list, args.clean, args.clean_all, args.reset]):
        parser.print_help()
        exit(0)
    
    # List files if requested
    if args.list:
        list_files_by_directory()
    
    # Clean up old files if requested
    if args.clean:
        deleted_count = cleanup_all_data_directories(
            max_age_days=args.max_age,
            keep_latest=args.keep_latest,
            dry_run=args.dry_run,
            force_clean=args.force
        )
        
        if args.dry_run:
            print(f"\nWould delete {deleted_count} files in total.")
        else:
            print(f"\nDeleted {deleted_count} files in total.")
    
    # Clean all output files if requested
    if args.clean_all:
        deleted_count = clean_all_output_files(dry_run=args.dry_run)
        
        if args.dry_run:
            print(f"\nWould delete ALL {deleted_count} output files.")
        else:
            print(f"\nDeleted ALL {deleted_count} output files.")
    
    # Reset all data, models, and visualizations if requested
    if args.reset:
        deleted_count = clean_all_data_models_visualizations(dry_run=args.dry_run)
        
        if args.dry_run:
            print(f"\nWould reset ALL {deleted_count} files (data, models, visualizations).")
        else:
            print(f"\nReset ALL {deleted_count} files (data, models, visualizations).")