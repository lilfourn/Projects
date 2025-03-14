#!/usr/bin/env python3
"""
Cleanup script to remove files that are no longer needed after the reorganization.
This script will remove:
1. Original files in the src directory that have been moved to subdirectories
2. Original files in the root directory that have been moved to subdirectories
3. Temporary wrapper files created during the migration

The script will ask for confirmation before deleting any files.
"""

import os
import sys
import shutil
from pathlib import Path

# Define the project root
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Files in src/ that have been moved to subdirectories
SRC_FILES_TO_REMOVE = [
    'src/getProjections.py',
    'src/data_processing.py',
    'src/data_cleanup.py',
    'src/data_quality.py',
    'src/data_quality_check_derived.py',
    'src/feature_engineering.py',
    'src/enhanced_defensive_features.py',
    'src/model_builder.py',
    'src/train_target_models.py',
    'src/predict.py',
    'src/ensemble_models.py',
    'src/feature_viz.py',
    'src/feature_importance_viz.py',
    'src/model_comparison.py',
    'src/config.py',
    'src/memory_utils.py',
    'src/run_pipeline.py',
]

# Files in root that have been moved to subdirectories
ROOT_FILES_TO_REMOVE = [
    'fetchGameData.py',
    'scrapeData.py',
    'train_all_targets.py',
    'train_xgboost_models.py',
    'predict.py',
    'analyze_projections.py',
]

# Temporary wrapper files created during migration
WRAPPER_FILES_TO_REMOVE = [
    'fetchGameData_wrapper.py',
    'scrapeData_wrapper.py',
    'train_all_targets_wrapper.py',
    'train_xgboost_models_wrapper.py',
    'predict_wrapper.py',
    'analyze_projections_wrapper.py',
]

def confirm_deletion(files):
    """Ask for confirmation before deleting files."""
    print("\nThe following files will be deleted:")
    for file in files:
        file_path = os.path.join(PROJECT_ROOT, file)
        if os.path.exists(file_path):
            print(f"  - {file}")
    
    while True:
        response = input("\nAre you sure you want to delete these files? (yes/no): ").lower()
        if response in ['yes', 'y']:
            return True
        elif response in ['no', 'n']:
            return False
        else:
            print("Please answer 'yes' or 'no'.")

def remove_files(files):
    """Remove the specified files."""
    removed_count = 0
    skipped_count = 0
    
    for file in files:
        file_path = os.path.join(PROJECT_ROOT, file)
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"Removed: {file}")
                removed_count += 1
            except Exception as e:
                print(f"Error removing {file}: {str(e)}")
                skipped_count += 1
        else:
            print(f"Skipped (not found): {file}")
            skipped_count += 1
    
    return removed_count, skipped_count

def main():
    """Main function to execute the cleanup process."""
    print("Starting cleanup of old files after reorganization...")
    
    # Combine all files to remove
    all_files_to_remove = SRC_FILES_TO_REMOVE + ROOT_FILES_TO_REMOVE + WRAPPER_FILES_TO_REMOVE
    
    # Filter out files that don't exist
    existing_files = [file for file in all_files_to_remove 
                     if os.path.exists(os.path.join(PROJECT_ROOT, file))]
    
    if not existing_files:
        print("No files found to remove. Cleanup already completed or files not found.")
        return
    
    # Ask for confirmation
    if confirm_deletion(existing_files):
        removed_count, skipped_count = remove_files(existing_files)
        print(f"\nCleanup completed: {removed_count} files removed, {skipped_count} files skipped.")
    else:
        print("\nCleanup cancelled. No files were removed.")

if __name__ == "__main__":
    main()
