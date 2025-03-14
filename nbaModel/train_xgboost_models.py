#!/usr/bin/env python3
"""
Script to train XGBoost models for all NBA target statistics
"""

import os
import sys
import logging
import argparse
import subprocess
import time
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def train_all_xgboost_models(
    targets=None,
    date_suffix=None,
    use_parallel=True,
    max_parallel=4,
    tune_hyperparams=True
):
    """
    Train XGBoost models for all target variables
    
    Args:
        targets (list, optional): List of target variables to train models for.
            If None, all default targets will be used.
        date_suffix (str, optional): Date suffix for model files. If None, today's date will be used.
        use_parallel (bool): Whether to train models in parallel
        max_parallel (int): Maximum number of parallel processes
        tune_hyperparams (bool): Whether to tune hyperparameters for each model
    """
    if date_suffix is None:
        # Use today's date as the suffix
        date_suffix = datetime.now().strftime("%Y%m%d")
    
    # Default targets if none provided
    if targets is None:
        targets = ['pts', 'reb', 'ast', 'fgm', 'fga', 'tptfgm', 'tptfga']
    
    logging.info(f"Training XGBoost models for targets: {targets}")
    
    # Path to the train_target_models.py script
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                              'src', 'train_target_models.py')
    
    if not os.path.exists(script_path):
        logging.error(f"Training script not found at {script_path}")
        return False
    
    # Track running processes
    processes = []
    completed_targets = []
    failed_targets = []
    
    for target in targets:
        # Build command for this target
        cmd = [
            sys.executable,
            script_path,
            "--targets", target,
            "--model-type", "xgboost"
        ]
        
        if tune_hyperparams:
            cmd.append("--optimize-features")
        
        logging.info(f"Starting training for target: {target}")
        
        if use_parallel:
            # Start the process
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            processes.append((target, proc))
            
            # If we've reached max_parallel, wait for one to complete
            if len(processes) >= max_parallel:
                logging.info(f"Reached max parallel processes ({max_parallel}), waiting for one to complete...")
                _wait_for_process_completion(processes, completed_targets, failed_targets)
        else:
            # Run sequentially
            try:
                logging.info(f"Running command: {' '.join(cmd)}")
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                logging.info(f"Training completed for target: {target}")
                completed_targets.append(target)
            except subprocess.CalledProcessError as e:
                logging.error(f"Error training model for target {target}: {e}")
                logging.error(f"STDOUT: {e.stdout}")
                logging.error(f"STDERR: {e.stderr}")
                failed_targets.append(target)
    
    # Wait for any remaining processes to complete
    while processes:
        _wait_for_process_completion(processes, completed_targets, failed_targets)
    
    # Report results
    logging.info(f"Training completed. Success: {len(completed_targets)}, Failed: {len(failed_targets)}")
    if completed_targets:
        logging.info(f"Successfully trained models for: {', '.join(completed_targets)}")
    if failed_targets:
        logging.error(f"Failed to train models for: {', '.join(failed_targets)}")
    
    return len(failed_targets) == 0

def _wait_for_process_completion(processes, completed_targets, failed_targets):
    """
    Wait for at least one process to complete
    
    Args:
        processes (list): List of (target, process) tuples
        completed_targets (list): List to append completed targets to
        failed_targets (list): List to append failed targets to
    """
    for i, (target, proc) in enumerate(processes):
        # Check if this process has completed
        if proc.poll() is not None:
            stdout, stderr = proc.communicate()
            if proc.returncode == 0:
                logging.info(f"Training completed for target: {target}")
                completed_targets.append(target)
            else:
                logging.error(f"Error training model for target {target}, return code {proc.returncode}")
                if stdout:
                    logging.debug(f"STDOUT: {stdout}")
                if stderr:
                    logging.error(f"STDERR: {stderr}")
                failed_targets.append(target)
            
            # Remove this process from the list
            processes.pop(i)
            return
    
    # If we get here, no process has completed yet, wait a bit
    time.sleep(2)

def main():
    """Parse command line arguments and train models"""
    parser = argparse.ArgumentParser(description="Train XGBoost models for all NBA target statistics")
    
    parser.add_argument("--targets", type=str, nargs="+",
                        help="Space-separated list of target variables to train models for")
    parser.add_argument("--date-suffix", type=str,
                        help="Date suffix for model files (default: today's date in YYYYMMDD format)")
    parser.add_argument("--sequential", action="store_true",
                        help="Train models sequentially instead of in parallel")
    parser.add_argument("--max-parallel", type=int, default=4,
                        help="Maximum number of parallel training processes (default: 4)")
    parser.add_argument("--no-hyperparameter-tuning", action="store_true",
                        help="Skip hyperparameter tuning for faster training")
    
    args = parser.parse_args()
    
    # Train models
    success = train_all_xgboost_models(
        targets=args.targets,
        date_suffix=args.date_suffix,
        use_parallel=not args.sequential,
        max_parallel=args.max_parallel,
        tune_hyperparams=not args.no_hyperparameter_tuning
    )
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()