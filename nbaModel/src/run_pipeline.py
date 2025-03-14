import os
import argparse
import logging
from datetime import datetime

# Import the progress tracking utilities
try:
    from src.memory_utils import PipelineTracker, ProgressLogger, TQDM_AVAILABLE
except ImportError:
    try:
        # Try importing without the src prefix
        from memory_utils import PipelineTracker, ProgressLogger, TQDM_AVAILABLE
    except ImportError:
        # If not available, define placeholders
        TQDM_AVAILABLE = False
        PipelineTracker = None
        ProgressLogger = None
        logging.warning("Progress tracking utilities not available")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Define the pipeline stages for the tracker
PIPELINE_STAGES = [
    "Data Processing",
    "Feature Engineering",
    "Model Training",
    "Model Evaluation",
    "Predictions",
    "Visualizations",
    "Cleanup"
]

# Global pipeline tracker instance
pipeline_tracker = None

def initialize_pipeline_tracker(use_tqdm=True):
    """Initialize the global pipeline tracker"""
    global pipeline_tracker
    if TQDM_AVAILABLE and PipelineTracker is not None:
        pipeline_tracker = PipelineTracker(stages=PIPELINE_STAGES, use_tqdm=use_tqdm)
    return pipeline_tracker

def run_data_processing(low_memory_mode=False, cleanup=False, max_age_days=30, keep_latest=3):
    """
    Run the data processing pipeline
    
    Args:
        low_memory_mode (bool): Whether to use aggressive memory-saving techniques
        cleanup (bool): Whether to clean up old data files after processing
        max_age_days (int): Maximum age in days to keep files during cleanup
        keep_latest (int): Minimum number of latest files to keep during cleanup
        
    Returns:
        bool: True if processing succeeded, False otherwise
    """
    try:
        from src.data_processing import main as process_data
        from src.config import cleanup_all_data_directories
    except ImportError:
        # Try importing without the src prefix
        from data_processing import main as process_data
        from config import cleanup_all_data_directories
    
    # Start data processing stage
    global pipeline_tracker
    if pipeline_tracker:
        pipeline_tracker.start_stage("Data Processing")
    
    logging.info("Starting data processing...")
    result = process_data(low_memory_mode=low_memory_mode)
    
    if result is not None:
        logging.info("Data processing completed successfully")
        
        # Run data cleanup if requested
        if cleanup:
            logging.info("Cleaning up old data files...")
            deleted_count = cleanup_all_data_directories(
                max_age_days=max_age_days,
                keep_latest=keep_latest,
                dry_run=False
            )
            logging.info(f"Deleted {deleted_count} old data files")
        
        # Complete the stage
        if pipeline_tracker:
            pipeline_tracker.complete_stage()
        
        return True
    else:
        logging.error("Data processing failed")
        
        # Mark stage as complete but with failure
        if pipeline_tracker:
            pipeline_tracker.complete_stage()
        
        return False

def run_feature_engineering(recache=False, remove_redundant=True, validate_derived=True):
    """
    Run the feature engineering pipeline
    
    Args:
        recache (bool): Whether to force regeneration of feature cache
        remove_redundant (bool): Whether to detect and remove redundant features
        validate_derived (bool): Whether to validate and fix derived features
    """
    import os
    from datetime import datetime
    import pandas as pd
    try:
        from src.feature_engineering import engineer_all_features
    except ImportError:
        # Try importing without the src prefix
        from feature_engineering import engineer_all_features
    
    # Start feature engineering stage
    global pipeline_tracker
    if pipeline_tracker:
        pipeline_tracker.start_stage("Feature Engineering")
    
    # Get the latest processed data file
    current_date = datetime.now().strftime("%Y%m%d")
    processed_dir = "/Users/lukesmac/Projects/nbaModel/data/processed"
    processed_data_path = os.path.join(processed_dir, f"processed_nba_data_{current_date}.csv")
    
    # If today's file doesn't exist, try to find the latest one
    if not os.path.exists(processed_data_path):
        if os.path.exists(processed_dir):
            processed_files = sorted(
                [f for f in os.listdir(processed_dir) if f.startswith("processed_nba_data_")],
                reverse=True
            )
            if processed_files:
                processed_data_path = os.path.join(processed_dir, processed_files[0])
            else:
                logging.error("No processed data file found")
                if pipeline_tracker:
                    pipeline_tracker.complete_stage()
                return False
    
    # Load the processed data
    try:
        processed_data = pd.read_csv(processed_data_path)
        logging.info(f"Loaded processed data with {processed_data.shape[0]} rows and {processed_data.shape[1]} columns")
    except Exception as e:
        logging.error(f"Error loading processed data: {str(e)}")
        if pipeline_tracker:
            pipeline_tracker.complete_stage()
        return False
    
    # Apply feature engineering
    logging.info("Starting feature engineering calculations...")
    try:
        # Run data quality checks on processed data
        try:
            try:
                from src.data_quality import run_all_quality_checks
            except ImportError:
                # Try importing without the src prefix
                from data_quality import run_all_quality_checks
                
            processed_data = run_all_quality_checks(
                processed_data, 
                check_derived=validate_derived
            )
            logging.info("Applied data quality checks to processed data")
        except ImportError:
            logging.warning("Data quality module not available, skipping checks")
        
        # Apply feature engineering with new optimization options
        engineered_data = engineer_all_features(
            processed_data,
            remove_unnecessary_columns=True,
            use_cache=not recache,
            remove_redundant_features=remove_redundant,
            validate_derived_values=validate_derived
        )
        
        # Save engineered data
        output_dir = "/Users/lukesmac/Projects/nbaModel/data/engineered"
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, f"engineered_nba_data_{current_date}.csv")
        engineered_data.to_csv(output_path, index=False)
        
        logging.info(f"Feature engineering completed successfully. Saved to {output_path}")
        
        # Report feature statistics
        initial_cols = processed_data.shape[1]
        final_cols = engineered_data.shape[1]
        added_cols = final_cols - initial_cols
        
        if remove_redundant:
            logging.info(f"Feature optimization: {added_cols} new features added, redundancy checks applied")
        
        if pipeline_tracker:
            pipeline_tracker.complete_stage()
        
        return True
    except Exception as e:
        logging.error(f"Error during feature engineering: {str(e)}")
        if pipeline_tracker:
            pipeline_tracker.complete_stage()
        return False

def run_model_training(model_type='random_forest', tune_hyperparams=False, use_cv=True, train_separate_models=False, 
                  low_resource_mode=False, update_feature_importance=True, importance_threshold=0.01):
    """
    Run the model training pipeline
    
    Args:
        model_type (str): Type of model to train ('decision_tree', 'random_forest', 'gradient_boosting')
        tune_hyperparams (bool): Whether to tune hyperparameters
        use_cv (bool): Whether to use cross-validation
        train_separate_models (bool): Whether to train separate models for each stat
        low_resource_mode (bool): Whether to use lower resource settings (fewer estimators, reduced depth)
        update_feature_importance (bool): Whether to update feature importance after training
        importance_threshold (float): Minimum threshold for feature importance
    """
    try:
        from src.model_builder import train_and_evaluate_model
    except ImportError:
        # Try importing without the src prefix
        from model_builder import train_and_evaluate_model
    import os
    
    # Start model training stage
    global pipeline_tracker
    if pipeline_tracker:
        pipeline_tracker.start_stage("Model Training")
    
    logging.info(f"Starting model training with {model_type} model...")
    try:
        # Pass the low_resource_mode parameter
        model, metrics = train_and_evaluate_model(
            model_type=model_type,
            tune_hyperparams=tune_hyperparams,
            use_cv=use_cv,
            train_separate_models=train_separate_models,
            low_resource_mode=low_resource_mode
        )
        
        if model is not None and metrics is not None:
            logging.info("Model training completed successfully")
            
            # Move to evaluation stage for feature importance
            if pipeline_tracker:
                pipeline_tracker.complete_stage()
                pipeline_tracker.start_stage("Model Evaluation")
            
            # Update feature importance if requested
            if update_feature_importance:
                logging.info("Updating feature importance from trained model...")
                try:
                    try:
                        from src.feature_engineering import update_feature_importance_from_model
                    except ImportError:
                        # Try importing without the src prefix
                        from feature_engineering import update_feature_importance_from_model
                    
                    # Find the latest model file in the models directory
                    models_dir = "/Users/lukesmac/Projects/nbaModel/models"
                    if os.path.exists(models_dir):
                        model_files = sorted(
                            [f for f in os.listdir(models_dir) if f.endswith('.joblib')],
                            reverse=True
                        )
                        
                        if model_files:
                            model_file = os.path.join(models_dir, model_files[0])
                            logging.info(f"Using model file: {model_file}")
                            
                            # Update feature importance from the model
                            updated = update_feature_importance_from_model(
                                model_file=model_file,
                                threshold=importance_threshold
                            )
                            
                            if updated:
                                num_features = len(updated)
                                logging.info(f"Updated feature importance with {num_features} features")
                                # Show top 10 features
                                top_features = sorted(updated.items(), key=lambda x: x[1], reverse=True)[:10]
                                logging.info("Top 10 most important features:")
                                for feature, importance in top_features:
                                    logging.info(f"  {feature}: {importance:.4f}")
                            else:
                                logging.warning("Failed to update feature importance")
                        else:
                            logging.warning("No model files found in models directory")
                    else:
                        logging.warning(f"Models directory not found: {models_dir}")
                except Exception as e:
                    logging.error(f"Error updating feature importance: {str(e)}")
            
            # Complete evaluation stage
            if pipeline_tracker:
                pipeline_tracker.complete_stage()
            
            return True
        else:
            logging.error("Model training failed")
            
            # Complete both stages with failure
            if pipeline_tracker:
                pipeline_tracker.complete_stage()  # Complete training stage
                if update_feature_importance:
                    pipeline_tracker.start_stage("Model Evaluation")
                    pipeline_tracker.complete_stage()  # Complete evaluation stage
            
            return False
    
    except Exception as e:
        logging.error(f"Error during model training: {str(e)}")
        
        # Complete training stage with failure
        if pipeline_tracker:
            pipeline_tracker.complete_stage()
        
        return False

def run_predictions(player_names=None, team=None, opponent=None, batch_size=10, 
                 use_optimized_types=True, optimize_memory=True):
    """
    Run predictions for one or more players with memory optimization
    
    Args:
        player_names (list): List of player names
        team (str): Team abbreviation
        opponent (str): Opponent team abbreviation
        batch_size (int): Number of players to process in each batch
        use_optimized_types (bool): Whether to use optimized data types
        optimize_memory (bool): Whether to use memory optimization techniques
    """
    # Start predictions stage
    global pipeline_tracker
    if pipeline_tracker:
        pipeline_tracker.start_stage("Predictions", total_items=len(player_names) if player_names else 0, unit="player")
    
    if optimize_memory:
        try:
            from src.predict import predict_multiple_players, format_prediction_output
        except ImportError:
            # Try importing without the src prefix
            from predict import predict_multiple_players, format_prediction_output
        
        try:
            from src.memory_utils import memory_usage_report, progress_map
        except ImportError:
            # Try importing without the src prefix
            from memory_utils import memory_usage_report, progress_map
    else:
        try:
            from src.predict import predict_player_performance, format_prediction_output
        except ImportError:
            # Try importing without the src prefix
            from predict import predict_player_performance, format_prediction_output
        
        try:
            from src.memory_utils import progress_map
        except ImportError:
            try:
                # Try importing without the src prefix
                from memory_utils import progress_map
            except ImportError:
                progress_map = None
    
    if not player_names:
        logging.error("No player names provided")
        if pipeline_tracker:
            pipeline_tracker.complete_stage()
        return False
    
    logging.info(f"Running predictions for {len(player_names)} players...")
    
    # Report initial memory usage
    if optimize_memory:
        initial_mem = memory_usage_report()
        logging.info(f"Initial memory usage: {initial_mem:.2f} MB")
    
    success = False
    
    try:
        # Use optimized batch prediction if requested and more than one player
        if optimize_memory and len(player_names) > 1:
            # Convert player names to list of player dicts
            players = [{'name': name, 'team': team} for name in player_names]
            
            # Use batch prediction
            predictions = predict_multiple_players(
                player_list=players,
                opponent=opponent,
                batch_size=batch_size,
                use_optimized_types=use_optimized_types
            )
            
            # Print the results
            if predictions:
                for prediction in predictions:
                    print(format_prediction_output(prediction))
                    success = True
                    if pipeline_tracker and pipeline_tracker.stage_progress:
                        pipeline_tracker.update_stage(1)
        else:
            # Use regular prediction for each player
            try:
                from src.predict import predict_player_performance
            except ImportError:
                # Try importing without the src prefix
                from predict import predict_player_performance
            
            # Prediction function for mapping
            def predict_for_player(player_name):
                try:
                    prediction = predict_player_performance(
                        player_name=player_name, 
                        team=team, 
                        opponent=opponent
                    )
                    
                    if prediction:
                        print(format_prediction_output(prediction))
                        return prediction
                    else:
                        logging.warning(f"Failed to generate prediction for {player_name}")
                        return None
                except Exception as e:
                    logging.error(f"Error predicting for {player_name}: {str(e)}")
                    return None
            
            # Use progress_map if available
            if progress_map:
                results = progress_map(
                    predict_for_player, 
                    player_names, 
                    desc="Predicting player performance", 
                    unit="player",
                    use_tqdm=False  # Don't use internal tqdm since we have the pipeline tracker
                )
                success = any(result is not None for result in results)
            else:
                # Fall back to regular loop
                for player_name in player_names:
                    logging.info(f"Predicting performance for {player_name}...")
                    prediction = predict_for_player(player_name)
                    if prediction:
                        success = True
                    
                    if pipeline_tracker and pipeline_tracker.stage_progress:
                        pipeline_tracker.update_stage(1)
        
        if success:
            logging.info("Predictions completed successfully")
            if pipeline_tracker:
                pipeline_tracker.complete_stage()
            return True
        else:
            logging.error("All predictions failed")
            if pipeline_tracker:
                pipeline_tracker.complete_stage()
            return False
            
    except Exception as e:
        logging.error(f"Error during predictions: {str(e)}")
        if pipeline_tracker:
            pipeline_tracker.complete_stage()
        return False

def run_visualizations(show_plots=False):
    """
    Run the feature importance visualization tool
    
    Args:
        show_plots (bool): Whether to display plots (in addition to saving them)
        
    Returns:
        bool: True if visualization was successful, False otherwise
    """
    # Start visualizations stage
    global pipeline_tracker
    if pipeline_tracker:
        pipeline_tracker.start_stage("Visualizations")
    
    try:
        try:
            from src.feature_viz import create_all_visualizations
        except ImportError:
            # Try importing without the src prefix
            from feature_viz import create_all_visualizations
        
        # Run all visualizations
        logging.info("Generating feature importance visualizations...")
        result = create_all_visualizations(show_plots=show_plots)
        
        if result:
            logging.info("Feature importance visualizations completed successfully")
            
            if pipeline_tracker:
                pipeline_tracker.complete_stage()
                
            return True
        else:
            logging.warning("Feature importance visualizations failed")
            
            if pipeline_tracker:
                pipeline_tracker.complete_stage()
                
            return False
            
    except Exception as e:
        logging.error(f"Error running visualizations: {str(e)}")
        
        if pipeline_tracker:
            pipeline_tracker.complete_stage()
            
        return False

def run_full_pipeline(model_type='random_forest', tune_hyperparams=False, use_cv=True, 
                  train_separate_models=False, player_names=None, team=None, opponent=None, 
                  low_resource_mode=False, cleanup=False, max_age_days=30, keep_latest=3,
                  update_feature_importance=True, importance_threshold=0.01, create_visualizations=True,
                  optimize_memory=True, use_optimized_types=True, batch_size=10, fresh_start=False,
                  recache_features=False, remove_redundant_features=True, validate_derived_features=True):
    """
    Run the full NBA model pipeline
    
    Args:
        model_type (str): Type of model to train ('decision_tree', 'random_forest', 'gradient_boosting')
        tune_hyperparams (bool): Whether to tune hyperparameters
        use_cv (bool): Whether to use cross-validation
        train_separate_models (bool): Whether to train separate models for each stat
        player_names (list): List of player names for predictions
        team (str): Team abbreviation
        opponent (str): Opponent team abbreviation
        low_resource_mode (bool): Whether to use lower resource settings for training
        cleanup (bool): Whether to clean up old data files at the end of processing
        max_age_days (int): Maximum age in days to keep files during cleanup
        keep_latest (int): Minimum number of latest files to keep during cleanup
        update_feature_importance (bool): Whether to update feature importance after training
        importance_threshold (float): Minimum threshold for feature importance
        create_visualizations (bool): Whether to create feature importance visualizations
        optimize_memory (bool): Whether to use memory optimization techniques during prediction
        use_optimized_types (bool): Whether to use optimized data types (float32 instead of float64, etc.)
        batch_size (int): Number of players to process in each batch during predictions
        fresh_start (bool): Whether to delete all output files before starting the pipeline
        recache_features (bool): Whether to force regeneration of feature cache
        remove_redundant_features (bool): Whether to detect and remove redundant features
        validate_derived_features (bool): Whether to validate and fix derived features
    """
    # Initialize the pipeline tracker
    global pipeline_tracker
    pipeline_tracker = initialize_pipeline_tracker()
    
    logging.info(f"Starting full NBA model pipeline with {model_type} model...")
    
    # Clean all output files if fresh_start is requested
    if fresh_start:
        logging.info("Fresh start requested - cleaning all output files")
        try:
            try:
                from src.data_cleanup import clean_all_output_files
            except ImportError:
                # Try importing without the src prefix
                from data_cleanup import clean_all_output_files
            deleted_count = clean_all_output_files(dry_run=False)
            logging.info(f"Deleted {deleted_count} output files to start fresh")
        except Exception as e:
            logging.error(f"Error cleaning output files: {str(e)}")
    
    # Step 1: Data Processing
    process_success = run_data_processing(
        low_memory_mode=low_resource_mode,
        cleanup=False  # Don't clean up yet, do it at the end of the pipeline
    )
    if not process_success:
        logging.error("Pipeline stopped at data processing step")
        if pipeline_tracker:
            pipeline_tracker.finish(success=False)
        return False
    
    # Step 2: Feature Engineering with enhanced options
    feature_success = run_feature_engineering(
        recache=recache_features,
        remove_redundant=remove_redundant_features,
        validate_derived=validate_derived_features
    )
    if not feature_success:
        logging.error("Pipeline stopped at feature engineering step")
        if pipeline_tracker:
            pipeline_tracker.finish(success=False)
        return False
    
    # Step 3: Model Training with advanced options
    model_success = run_model_training(
        model_type=model_type,
        tune_hyperparams=tune_hyperparams,
        use_cv=use_cv,
        train_separate_models=train_separate_models,
        low_resource_mode=low_resource_mode,
        update_feature_importance=update_feature_importance,
        importance_threshold=importance_threshold
    )
    if not model_success:
        logging.error("Pipeline stopped at model training step")
        if pipeline_tracker:
            pipeline_tracker.finish(success=False)
        return False
    
    # Step 4: Predictions (optional)
    if player_names:
        pred_success = run_predictions(
            player_names=player_names, 
            team=team, 
            opponent=opponent,
            batch_size=batch_size,
            use_optimized_types=use_optimized_types,
            optimize_memory=optimize_memory
        )
        if not pred_success:
            logging.warning("Predictions step failed")
            # Continue with the rest of the pipeline
    else:
        # Skip predictions stage in the tracker
        if pipeline_tracker:
            pipeline_tracker.start_stage("Predictions")
            pipeline_tracker.complete_stage()
    
    # Step 5: Feature Importance Visualizations (optional)
    if update_feature_importance and create_visualizations:
        run_visualizations(show_plots=False)
    else:
        # Skip visualizations stage in the tracker
        if pipeline_tracker:
            pipeline_tracker.start_stage("Visualizations")
            pipeline_tracker.complete_stage()
    
    # Step 6: Data Cleanup (optional)
    if pipeline_tracker:
        pipeline_tracker.start_stage("Cleanup")
        
    if cleanup:
        try:
            try:
                from src.data_cleanup import cleanup_all_data_directories
            except ImportError:
                # Try importing without the src prefix
                from data_cleanup import cleanup_all_data_directories
            deleted_count = cleanup_all_data_directories(
                max_age_days=max_age_days,
                keep_latest=keep_latest,
                dry_run=False,
                force_clean=False  # Only delete old files, keep the latest ones
            )
            logging.info(f"Deleted {deleted_count} old data files")
        except Exception as e:
            logging.warning(f"Error during cleanup: {str(e)}")
    
    if pipeline_tracker:
        pipeline_tracker.complete_stage()
        pipeline_tracker.finish(success=True)
    
    logging.info("Full NBA model pipeline completed successfully")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the NBA player performance prediction pipeline')
    
    # Pipeline stages
    parser.add_argument('--process-data', action='store_true', help='Run data processing')
    parser.add_argument('--engineer-features', action='store_true', help='Run feature engineering')
    parser.add_argument('--train-model', action='store_true', help='Train the model')
    parser.add_argument('--predict', action='store_true', help='Run predictions')
    parser.add_argument('--visualize', action='store_true', help='Generate feature importance visualizations')
    parser.add_argument('--full-pipeline', action='store_true', help='Run the full pipeline')
    parser.add_argument('--cleanup', action='store_true', help='Clean up old data files')
    parser.add_argument('--fresh-start', action='store_true', help='Delete all output files before starting the pipeline')
    
    # Model configuration options
    parser.add_argument('--model-type', type=str, default='random_forest', 
                      choices=['decision_tree', 'random_forest', 'gradient_boosting'],
                      help='Model type to use')
    parser.add_argument('--tune-hyperparams', action='store_true', help='Tune model hyperparameters')
    parser.add_argument('--use-cv', action='store_true', default=True, 
                      help='Use cross-validation for training')
    parser.add_argument('--no-cv', action='store_false', dest='use_cv',
                      help='Disable cross-validation (use simple train/test split)')
    parser.add_argument('--separate-models', action='store_true', 
                      help='Train separate models for each statistic')
    parser.add_argument('--low-resource-mode', action='store_true',
                      help='Use lower resource settings to reduce CPU and memory usage')
    
    # Feature engineering options
    feature_engineering_group = parser.add_argument_group('Feature Engineering Options')
    feature_engineering_group.add_argument('--recache-features', action='store_true',
                        help='Force regeneration of feature cache')
    feature_engineering_group.add_argument('--remove-redundant-features', action='store_true', default=True,
                        help='Detect and remove redundant features (default: True)')
    feature_engineering_group.add_argument('--no-remove-redundant', action='store_false', dest='remove_redundant_features',
                        help='Skip redundant feature detection and removal')
    feature_engineering_group.add_argument('--validate-derived-features', action='store_true', default=True,
                        help='Validate and fix derived features (default: True)')
    feature_engineering_group.add_argument('--no-validate-derived', action='store_false', dest='validate_derived_features',
                        help='Skip validation of derived features')
    
    # Feature importance options
    feature_importance_group = parser.add_argument_group('Feature Importance Options')
    feature_importance_group.add_argument('--update-importance', action='store_true', default=True,
                        help='Update feature importance after model training (default: True)')
    feature_importance_group.add_argument('--no-update-importance', action='store_false', dest='update_importance',
                        help='Skip updating feature importance after model training')
    feature_importance_group.add_argument('--importance-threshold', type=float, default=0.01,
                        help='Minimum importance threshold for including features (default: 0.01)')
    feature_importance_group.add_argument('--create-visualizations', action='store_true', default=True,
                        help='Create feature importance visualizations after training (default: True)')
    feature_importance_group.add_argument('--no-visualizations', action='store_false', dest='create_visualizations',
                        help='Skip creating feature importance visualizations')
    feature_importance_group.add_argument('--show-plots', action='store_true',
                        help='Display visualization plots (not just save to files)')
                        
    # Memory optimization options
    memory_group = parser.add_argument_group('Memory Optimization Options')
    memory_group.add_argument('--optimize-memory', action='store_true', default=True,
                        help='Use memory optimization techniques (default: True)')
    memory_group.add_argument('--no-optimize-memory', action='store_false', dest='optimize_memory',
                        help='Disable memory optimizations')
    memory_group.add_argument('--use-optimized-types', action='store_true', default=True,
                        help='Use optimized data types for memory efficiency (default: True)')
    memory_group.add_argument('--no-optimized-types', action='store_false', dest='use_optimized_types',
                        help='Disable optimized data types')
    memory_group.add_argument('--batch-size', type=int, default=10,
                        help='Number of players to process in each batch during predictions (default: 10)')
    memory_group.add_argument('--profile-memory', action='store_true',
                        help='Log detailed memory usage information during execution')
    
    # Progress tracking options
    progress_group = parser.add_argument_group('Progress Tracking Options')
    progress_group.add_argument('--progress-bar', action='store_true', default=True,
                       help='Show progress bars for pipeline stages (default: True)')
    progress_group.add_argument('--no-progress-bar', action='store_false', dest='progress_bar',
                       help='Disable progress bars, use only logging')
    
    # Prediction options
    parser.add_argument('--player-names', nargs='+', help='Player names for predictions')
    parser.add_argument('--team', type=str, help='Team abbreviation')
    parser.add_argument('--opponent', type=str, help='Opponent team abbreviation')
    
    # Data cleanup options
    parser.add_argument('--max-age', type=int, default=30, 
                      help='Maximum age in days to keep files during cleanup (default: 30)')
    parser.add_argument('--keep-latest', type=int, default=3, 
                      help='Minimum number of latest files to keep during cleanup (default: 3)')
    parser.add_argument('--cleanup-dry-run', action='store_true',
                      help='Show what would be deleted without actually deleting')
    parser.add_argument('--list-files', action='store_true',
                      help='List files in data directories without processing or cleaning')
    
    args = parser.parse_args()
    
    # Check if we should just list files
    if args.list_files:
        try:
            from src.data_cleanup import list_files_by_directory
        except ImportError:
            # Try importing without the src prefix
            from data_cleanup import list_files_by_directory
        list_files_by_directory()
        exit(0)
    
    # Check if we should just run cleanup
    if args.cleanup and not any([args.process_data, args.engineer_features, args.train_model, args.predict, args.full_pipeline]):
        try:
            from src.config import cleanup_all_data_directories
        except ImportError:
            # Try importing without the src prefix
            from config import cleanup_all_data_directories
        cleanup_all_data_directories(
            max_age_days=args.max_age,
            keep_latest=args.keep_latest,
            dry_run=args.cleanup_dry_run
        )
        exit(0)
    
    # Check if at least one stage was specified
    if not any([args.process_data, args.engineer_features, args.train_model, args.predict, args.visualize, args.full_pipeline, args.cleanup]):
        parser.print_help()
        exit(1)
    
    # Initialize progress tracking (global variable already declared at module level)
    if args.progress_bar:
        initialize_pipeline_tracker(use_tqdm=True)
    
    # Configure memory profiling
    if args.profile_memory:
        try:
            from src.memory_utils import memory_usage_report
        except ImportError:
            # Try importing without the src prefix
            from memory_utils import memory_usage_report
        initial_mem = memory_usage_report()
        logging.info(f"Initial memory usage: {initial_mem:.2f} MB")
    
    if args.full_pipeline:
        run_full_pipeline(
            model_type=args.model_type,
            tune_hyperparams=args.tune_hyperparams,
            use_cv=args.use_cv,
            train_separate_models=args.separate_models,
            player_names=args.player_names,
            team=args.team,
            opponent=args.opponent,
            low_resource_mode=args.low_resource_mode,
            cleanup=args.cleanup,
            max_age_days=args.max_age,
            keep_latest=args.keep_latest,
            update_feature_importance=args.update_importance,
            importance_threshold=args.importance_threshold,
            create_visualizations=args.create_visualizations,
            optimize_memory=args.optimize_memory,
            use_optimized_types=args.use_optimized_types,
            batch_size=args.batch_size,
            fresh_start=args.fresh_start,
            recache_features=args.recache_features,
            remove_redundant_features=args.remove_redundant_features,
            validate_derived_features=args.validate_derived_features
        )
    else:
        if args.process_data:
            run_data_processing(
                low_memory_mode=args.low_resource_mode,
                cleanup=args.cleanup,
                max_age_days=args.max_age,
                keep_latest=args.keep_latest
            )
        
        if args.engineer_features:
            run_feature_engineering(
                recache=args.recache_features,
                remove_redundant=args.remove_redundant_features,
                validate_derived=args.validate_derived_features
            )
        
        if args.train_model:
            run_model_training(
                model_type=args.model_type,
                tune_hyperparams=args.tune_hyperparams,
                use_cv=args.use_cv,
                train_separate_models=args.separate_models,
                low_resource_mode=args.low_resource_mode,
                update_feature_importance=args.update_importance,
                importance_threshold=args.importance_threshold
            )
        
        if args.predict:
            if not args.player_names:
                logging.error("Player names must be provided for predictions")
            else:
                run_predictions(
                    player_names=args.player_names,
                    team=args.team,
                    opponent=args.opponent,
                    batch_size=args.batch_size,
                    use_optimized_types=args.use_optimized_types,
                    optimize_memory=args.optimize_memory
                )
                
        if args.visualize:
            run_visualizations(show_plots=args.show_plots)
        
        # Clean up the pipeline tracker if individual operations were run
        if pipeline_tracker:
            pipeline_tracker.finish(success=True)
            
    # Report final memory usage if profiling
    if args.profile_memory:
        try:
            from src.memory_utils import memory_usage_report
        except ImportError:
            # Try importing without the src prefix
            from memory_utils import memory_usage_report
        final_mem = memory_usage_report()
        logging.info(f"Final memory usage: {final_mem:.2f} MB (Change: {final_mem - initial_mem:.2f} MB)")