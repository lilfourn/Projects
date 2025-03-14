#!/usr/bin/env python3
"""
Main entry point for the NBA model application.
This script provides a unified interface to run different components of the NBA model.
"""

import argparse
import sys
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main():
    """Main function to parse arguments and run the appropriate component."""
    parser = argparse.ArgumentParser(description='NBA Model - Main Entry Point')
    parser.add_argument('component', choices=[
        'fetch_game_data', 
        'scrape_data', 
        'get_projections',
        'process_data',
        'train_models',
        'predict',
        'visualize',
        'run_pipeline'
    ], help='Component to run')
    
    # Add common arguments
    parser.add_argument('--output-dir', type=str, default=None, 
                        help='Output directory for data files')
    parser.add_argument('--start-season', type=int, default=2024, 
                        help='Starting season year (e.g., 2024 for 2024-25 season)')
    parser.add_argument('--end-season', type=int, default=2025, 
                        help='Ending season year (e.g., 2025 for 2024-25 season)')
    
    # Parse known args first to get the component
    args, remaining_args = parser.parse_known_args()
    
    # Import the appropriate module based on the component
    if args.component == 'fetch_game_data':
        from src.data_collection.fetchGameData import main as component_main
        sys.argv = [sys.argv[0]] + remaining_args
        component_main()
    
    elif args.component == 'scrape_data':
        from src.data_collection.scrapeData import main as component_main
        sys.argv = [sys.argv[0]] + remaining_args
        component_main()
    
    elif args.component == 'get_projections':
        from src.data_collection.getProjections import main as component_main
        sys.argv = [sys.argv[0]] + remaining_args
        component_main()
    
    elif args.component == 'process_data':
        from src.data_processing.data_processing import main as component_main
        sys.argv = [sys.argv[0]] + remaining_args
        component_main()
    
    elif args.component == 'train_models':
        # Add train_models functionality directly in main.py
        from datetime import datetime
        import logging
        import subprocess
        import sys
        import os
        from src.utils.config import MODELS_DIR, DATA_DIR
        
        # Create a function to handle train_models
        def train_models_main():
            # Parse arguments specific to model training
            train_parser = argparse.ArgumentParser(description='Train NBA prediction models')
            train_parser.add_argument('--targets', type=str, nargs='+', default=None,
                                help='Target variables to train models for (e.g., pts reb ast)')
            train_parser.add_argument('--model-type', type=str, default='random_forest',
                                choices=['decision_tree', 'random_forest', 'gradient_boosting', 'xgboost'],
                                help='Type of model to train')
            train_parser.add_argument('--sequential', action='store_true',
                                help='Train models sequentially to avoid memory issues')
            train_parser.add_argument('--process-data', action='store_true',
                                help='Process data before training')
            train_parser.add_argument('--use-ensemble', action='store_true',
                                help='Use ensemble stacking')
            
            train_args = train_parser.parse_args(remaining_args)
            
            # Target variables to train models for
            TARGET_VARIABLES = ['pts', 'reb', 'ast', 'fgm', 'fga', 'tptfgm', 'tptfga']
            
            # Use specified targets if provided
            if train_args.targets:
                targets = train_args.targets
                logging.info(f"Using specified targets: {targets}")
            else:
                targets = TARGET_VARIABLES
                logging.info(f"Using all targets: {targets}")
            
            start_time = datetime.now()
            logging.info(f"Starting model training for all targets at {start_time}")
            
            if train_args.sequential:
                success = True
                # Train each target individually
                for target in targets:
                    logging.info(f"\n{'='*60}\nTraining model for {target}\n{'='*60}")
                    
                    # Build command for single target
                    cmd = [sys.executable, "-m", "src.models.train_target_models"]
                    cmd.extend(["--targets", target])
                    
                    if train_args.process_data:
                        cmd.append("--process-data")
                    
                    if train_args.model_type:
                        cmd.extend(["--model-type", train_args.model_type])
                    
                    # Run the command
                    logging.info(f"Running command: {' '.join(cmd)}")
                    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
                    
                    if result.returncode == 0:
                        logging.info(f"Model training for {target} completed successfully")
                    else:
                        logging.error(f"Model training for {target} failed with exit code {result.returncode}")
                        success = False
            else:
                # Train all targets at once
                cmd = [sys.executable, "-m", "src.models.train_target_models"]
                cmd.extend(["--targets"] + targets)
                
                if train_args.process_data:
                    cmd.append("--process-data")
                
                if train_args.model_type:
                    cmd.extend(["--model-type", train_args.model_type])
                
                # Run the command
                logging.info(f"Running command: {' '.join(cmd)}")
                result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
                
                if result.returncode == 0:
                    logging.info(f"Model training for all targets completed successfully")
                    success = True
                else:
                    logging.error(f"Model training failed with exit code {result.returncode}")
                    success = False
            
            end_time = datetime.now()
            logging.info(f"Training completed in {end_time - start_time}")
            
            if success:
                logging.info("All models trained successfully")
            else:
                logging.error("Some models failed to train")
            
            return success
        
        # Run the train_models functionality
        train_models_main()
    
    elif args.component == 'predict':
        from src.models.predict import main as component_main
        sys.argv = [sys.argv[0]] + remaining_args
        component_main()
    
    elif args.component == 'visualize':
        from src.visualization.feature_importance_viz import main as component_main
        sys.argv = [sys.argv[0]] + remaining_args
        component_main()
    
    elif args.component == 'run_pipeline':
        from src.utils.run_pipeline import run_full_pipeline
        # Parse additional arguments specific to the pipeline
        pipeline_parser = argparse.ArgumentParser(description='NBA Model Pipeline')
        pipeline_parser.add_argument('--model-type', type=str, default='random_forest',
                                    choices=['decision_tree', 'random_forest', 'gradient_boosting', 'xgboost'],
                                    help='Type of model to train')
        pipeline_parser.add_argument('--tune-hyperparams', action='store_true',
                                    help='Tune model hyperparameters')
        pipeline_parser.add_argument('--use-cv', action='store_true', default=True,
                                    help='Use cross-validation for model evaluation')
        pipeline_parser.add_argument('--train-separate-models', action='store_true',
                                    help='Train separate models for each stat')
        pipeline_parser.add_argument('--low-resource-mode', action='store_true',
                                    help='Use lower resource settings for training')
        pipeline_parser.add_argument('--cleanup', action='store_true',
                                    help='Clean up old data files after processing')
        pipeline_parser.add_argument('--create-visualizations', action='store_true', default=True,
                                    help='Create feature importance visualizations')
        
        pipeline_args = pipeline_parser.parse_args(remaining_args)
        
        # Run the pipeline with the parsed arguments
        run_full_pipeline(
            model_type=pipeline_args.model_type,
            tune_hyperparams=pipeline_args.tune_hyperparams,
            use_cv=pipeline_args.use_cv,
            train_separate_models=pipeline_args.train_separate_models,
            low_resource_mode=pipeline_args.low_resource_mode,
            cleanup=pipeline_args.cleanup,
            create_visualizations=pipeline_args.create_visualizations
        )

if __name__ == '__main__':
    main()
