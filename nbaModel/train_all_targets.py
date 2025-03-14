#!/usr/bin/env python3
"""
Script to train models for all target variables

This script will train models for all the target variables:
- pts: Points scored
- reb: Rebounds
- ast: Assists
- fgm: Field goals made
- fga: Field goals attempted
- tptfgm: Three-point field goals made
- tptfga: Three-point field goals attempted
"""

import os
import sys
import logging
import subprocess
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Target variables to train models for
TARGET_VARIABLES = ['pts', 'reb', 'ast', 'fgm', 'fga', 'tptfgm', 'tptfga']

def train_all_models(use_ensemble=False, use_time_series=True, process_data=True, sequential=False, model_type="random_forest"):
    """
    Train models for all target variables
    
    Args:
        use_ensemble (bool): Whether to use ensemble stacking (default: False)
        use_time_series (bool): Whether to use time series cross-validation
        process_data (bool): Whether to process data before training
        sequential (bool): Whether to train models one by one to avoid memory issues
        
    Returns:
        bool: Whether all training was successful
    """
    start_time = datetime.now()
    logging.info(f"Starting model training for all targets at {start_time}")
    
    if sequential:
        success = True
        # Train each target individually
        for target in TARGET_VARIABLES:
            logging.info(f"\n{'='*60}\nTraining model for {target}\n{'='*60}")
            
            # Build command for single target
            cmd = [sys.executable, "-m", "src.train_target_models"]
            cmd.extend(["--targets", target])
            
            # Add options
            if use_ensemble:
                cmd.append("--ensemble")
            if not use_time_series:
                cmd.append("--no-time-series")
            if process_data and target == TARGET_VARIABLES[0]:  # Only process data for first target
                cmd.append("--process-data")
            
            # Add model type
            cmd.extend(["--model-type", model_type])
            
            # Log the command
            logging.info(f"Running command: {' '.join(cmd)}")
            
            try:
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                logging.info(f"Model training for {target} completed successfully")
            except subprocess.CalledProcessError as e:
                logging.error(f"Error training model for {target}: {e}")
                logging.error(f"Command error: {e.stderr}")
                success = False
        
        end_time = datetime.now()
        duration = end_time - start_time
        logging.info(f"Training completed in {duration}")
        return success
    else:
        # Train all targets at once
        # Build command for all targets
        cmd = [sys.executable, "-m", "src.train_target_models"]
        
        # Add targets
        cmd.extend(["--targets"] + TARGET_VARIABLES)
        
        # Add options
        if use_ensemble:
            cmd.append("--ensemble")
        if not use_time_series:
            cmd.append("--no-time-series")
        if process_data:
            cmd.append("--process-data")
            
        # Add model type
        cmd.extend(["--model-type", model_type])
        
        # Log the command
        logging.info(f"Running command: {' '.join(cmd)}")
        
        # Run the command
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logging.info("Command output:")
            for line in result.stdout.splitlines():
                logging.info(line)
            
            end_time = datetime.now()
            duration = end_time - start_time
            logging.info(f"Model training completed successfully in {duration}")
            return True
        except subprocess.CalledProcessError as e:
            logging.error(f"Error running command: {e}")
            logging.error(f"Command output: {e.stdout}")
            logging.error(f"Command error: {e.stderr}")
            return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train NBA prediction models for all target variables")
    
    parser.add_argument("--ensemble", action="store_true",
                        help="Enable ensemble stacking (Warning: May cause errors)")
    parser.add_argument("--no-time-series", action="store_true",
                        help="Disable time series cross-validation")
    process_group = parser.add_mutually_exclusive_group()
    process_group.add_argument("--process-data", action="store_true",
                            help="Process data before training (default is based on script default)")
    process_group.add_argument("--no-process-data", action="store_true",
                            help="Skip data processing before training")
                            
    parser.add_argument("--targets", type=str, nargs='+',
                        help="Specific targets to train (default: all targets)")
    parser.add_argument("--sequential", action="store_true",
                        help="Train models one by one (helps with memory issues)")
    parser.add_argument("--model-type", type=str, default="random_forest",
                        choices=["random_forest", "xgboost", "gradient_boosting", "decision_tree", "lightgbm", "stacked"],
                        help="Type of model to train (default: random_forest)")
    
    args = parser.parse_args()
    
    # Override target variables if specified
    if args.targets:
        # Update the global variable correctly
        globals()['TARGET_VARIABLES'] = args.targets
        logging.info(f"Using specified targets: {args.targets}")
    
    # Determine whether to process data
    process_data = True  # Default to processing data
    if args.no_process_data:
        process_data = False
    elif args.process_data:
        process_data = True
    
    success = train_all_models(
        use_ensemble=args.ensemble,
        use_time_series=not args.no_time_series,
        process_data=process_data,
        sequential=args.sequential,
        model_type=args.model_type
    )
    
    if success:
        logging.info("All models trained successfully")
        sys.exit(0)
    else:
        logging.error("Error training models")
        sys.exit(1)