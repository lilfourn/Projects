#!/usr/bin/env python3
"""
Script to train individual models for specific target variables

This script is now located in the src directory and can be run as:
- From project root: python -m src.train_target_models
- From src directory: python train_target_models.py
- From anywhere with full path: python /path/to/src/train_target_models.py

Target variables:
- pts: Points scored
- reb: Rebounds
- ast: Assists
- fgm: Field goals made
- fga: Field goals attempted
- tptfgm: Three-point field goals made
- tptfga: Three-point field goals attempted
"""

import sys
import os

# Adjust Python path to allow running from both project root and src directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import os
import logging
import argparse
import pandas as pd
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Import model builder functions
try:
    # First try direct import (when running from src directory)
    try:
        from model_builder import (
            train_model, evaluate_model, analyze_feature_importance,
            load_training_data, create_feature_matrix, save_model, save_metrics
        )
    except ImportError:
        # Fall back to src prefix import (when running from project root)
        from src.model_builder import (
            train_model, evaluate_model, analyze_feature_importance,
            load_training_data, create_feature_matrix, save_model, save_metrics
        )
except ImportError:
    logging.error("Unable to import model_builder module. Make sure PYTHONPATH is set correctly.")
    exit(1)

# Import data processing and feature engineering if needed
try:
    # First try direct import (when running from src directory)
    try:
        from data_processing import main as process_data
        from feature_engineering import engineer_all_features
        data_pipeline_available = True
    except ImportError:
        # Fall back to src prefix import (when running from project root)
        from src.data_processing import main as process_data
        from src.feature_engineering import engineer_all_features
        data_pipeline_available = True
except ImportError:
    data_pipeline_available = False
    logging.warning("Data processing and feature engineering modules not available.")


def train_specific_target(target_name, use_ensemble=True, use_time_series=True, gap=3, test_size=0.2, 
                  remove_redundant_features=True, validate_features=True):
    """Train a model for a specific target variable
    
    Args:
        target_name (str): Name of the target variable (e.g., 'pts', 'reb', 'ast')
        use_ensemble (bool): Whether to use ensemble stacking
        use_time_series (bool): Whether to use time series cross-validation
        gap (int): Gap size for time series cross-validation
        test_size (float): Test size for train/test split
        remove_redundant_features (bool): Whether to detect and remove redundant features
        validate_features (bool): Whether to validate and fix derived features
        
    Returns:
        tuple: Model, metrics, feature importances
    """
    # Auto-disable ensemble stacking for newer target variables
    # which might cause issues with the current implementation
    new_targets = ['fgm', 'fga', 'tptfgm', 'tptfga']
    if target_name in new_targets and use_ensemble:
        logging.warning(f"Auto-disabling ensemble stacking for {target_name} to prevent errors")
        use_ensemble = False
    logging.info(f"Starting training process for target: {target_name}")
    
    # Load the training data
    data = load_training_data()
    if data is None:
        logging.error("Failed to load training data")
        return None, None, None
        
    # For the new target columns like fgm, fga, tptfgm, tptfga, we need to check if they're in the engineered data
    # If not, we should load them from the processed data and merge them
    if target_name in ['fgm', 'fga', 'tptfgm', 'tptfga'] and target_name not in data.columns:
        try:
            # Try to load the processed data which should have these columns
            processed_data_path = os.path.join("data", "processed", "processed_nba_data_20250313.csv")
            if os.path.exists(processed_data_path):
                logging.info(f"Loading target column {target_name} from processed data")
                processed_data = pd.read_csv(processed_data_path)
                
                # Check if the target column exists in the processed data
                if target_name in processed_data.columns:
                    # Add the target column to our data
                    data[target_name] = processed_data[target_name]
                    logging.info(f"Added {target_name} column from processed data")
                else:
                    logging.error(f"Target column {target_name} not found in processed data")
                    return None, None, None
            else:
                logging.error(f"Processed data file not found: {processed_data_path}")
                return None, None, None
        except Exception as e:
            logging.error(f"Error loading target column {target_name}: {str(e)}")
            return None, None, None
    
    logging.info(f"Loaded training data with {data.shape[0]} rows and {data.shape[1]} columns")
    
    # Apply data quality and feature optimization if requested
    if validate_features or remove_redundant_features:
        try:
            # First try direct import (when running from src directory)
            try:
                from feature_engineering import validate_derived_features, detect_and_handle_redundant_features, detect_unused_features
            except ImportError:
                # Fall back to src prefix import (when running from project root)
                from src.feature_engineering import validate_derived_features, detect_and_handle_redundant_features, detect_unused_features
            
            # First validate derived features if requested
            if validate_features:
                logging.info("Validating derived features...")
                data = validate_derived_features(data)
                logging.info("Feature validation complete")
            
            # Then remove redundant features if requested
            if remove_redundant_features:
                logging.info("Detecting and removing redundant features...")
                data = detect_and_handle_redundant_features(data)
                
                # Also remove unused features (those with very low importance)
                data = detect_unused_features(data)
                logging.info(f"Feature optimization complete, {data.shape[1]} columns remaining")
        except ImportError:
            logging.warning("Could not import feature engineering functions for optimization")
    
    # Create feature matrix manually to avoid issues with the create_feature_matrix function
    # Map target name to possible column names
    column_mapping = {
        'pts': ['pts', 'PTS'],
        'reb': ['reb', 'TRB', 'REB'],
        'ast': ['ast', 'AST'],
        'fgm': ['fgm', 'FGM', 'FG'],
        'fga': ['fga', 'FGA'],
        'tptfgm': ['tptfgm', '3PM', '3P'],
        'tptfga': ['tptfga', '3PA']
    }
    
    # Find the actual column name in the data
    actual_column = None
    if target_name in column_mapping:
        for col in column_mapping[target_name]:
            if col in data.columns:
                actual_column = col
                break
    
    if actual_column is None:
        if target_name in data.columns:
            actual_column = target_name
        else:
            logging.error(f"Target column '{target_name}' not found in data")
            logging.info(f"Available columns: {', '.join(data.columns)}")
            return None, None, None
    
    logging.info(f"Using column '{actual_column}' for target '{target_name}'")
    
    # Extract target column
    y = data[[actual_column]]
    
    # Create a list of columns to exclude from features
    exclude_cols = [
        # Target and actual column
        target_name, actual_column,
        
        # Statistics columns (lowercase and uppercase)
        'pts', 'PTS', 'reb', 'TRB', 'REB', 'ast', 'AST', 
        'stl', 'STL', 'blk', 'BLK', 'plusMinus', '+/-',
        'fgm', 'FGM', 'FG', 'fga', 'FGA', 'fgp', 'FG%',
        'tptfgm', '3PM', '3P', 'tptfga', '3PA', 'tptfgp', '3P%',
        'ftm', 'FTM', 'fta', 'FTA', 'ftp', 'FT%',
        
        # Metadata columns
        'gameID', 'playerID', 'teamID', 'longName', 
        'opponent', 'teamAbv', 'team', 'game_date'
    ]
    
    # Remove target columns and non-feature columns
    feature_cols = [col for col in data.columns if col not in exclude_cols]
    
    X = data[feature_cols].copy()
    
    # Handle missing values
    for col in X.columns:
        if X[col].dtype.kind in 'if':  # numeric columns (integer or float)
            mean_val = X[col].mean() if not X[col].isna().all() else 0
            X[col] = X[col].fillna(mean_val)
        else:
            # For non-numeric columns, fill with the most common value
            most_common = X[col].mode()[0] if not X[col].isna().all() else "unknown"
            X[col] = X[col].fillna(most_common)
    
    # Get feature names
    feature_names = X.columns.tolist()
    
    if X.shape[0] == 0 or y.shape[0] == 0:
        logging.error(f"No valid data found for target: {target_name}")
        return None, None, None
    
    logging.info(f"Created feature matrix with {X.shape[0]} rows and {X.shape[1]} columns")
    logging.info(f"Target variable: {target_name}")
    
    # Train model
    model_result = train_model(
        X, y, 
        model_type='random_forest',
        use_ensemble_stacking=use_ensemble
    )
    
    if model_result is None:
        logging.error(f"Failed to train model for target: {target_name}")
        return None, None, None
        
    # Unpack the result tuple - (model, feature_importance)
    model, model_importance = model_result
    
    # Evaluate model
    metrics = evaluate_model(
        model, X, y, 
        feature_names=feature_names,
        time_series_validation=use_time_series,
        time_gap=gap,
        cv=5  # Use 5-fold cross-validation
    )
    
    # Get feature importance
    importance = analyze_feature_importance(
        model, X, y,
        feature_names=feature_names,
        permutation_importance_only=True
    )
    
    # Save the model, metrics, and feature importance
    date_str = datetime.now().strftime("%Y%m%d")
    
    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)
    
    # Create a mapping from actual column name to standardized target name
    target_mapping = {}
    if 'actual_column' in locals() and actual_column != target_name:
        target_mapping[actual_column] = target_name
        logging.info(f"Mapped {actual_column} to {target_name} for model output")
    
    # Save model
    model_dir = f"models/nba_{target_name}_model_{date_str}.joblib"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"nba_dt_model_{date_str}.joblib")
    save_model(model, model_path)
    logging.info(f"Saved model to {model_path}")
    
    # Save metrics
    metrics_path = f"models/nba_{target_name}_metrics_{date_str}.json"
    # Add target mapping to metrics if needed
    if target_mapping:
        metrics['target_mapping'] = target_mapping
    
    # Add feature names to metrics
    metrics['feature_names'] = feature_cols
    metrics['target_names'] = [actual_column]
    
    # Save metrics manually as the save_metrics function needs target_names parameter
    with open(metrics_path, 'w') as f:
        import json
        json.dump(metrics, f, indent=2)
    logging.info(f"Saved metrics to {metrics_path}")
    
    # Save feature importance
    importance_path = f"models/feature_importance_{target_name}_{date_str}.json"
    with open(importance_path, 'w') as f:
        import json
        json.dump(importance, f, indent=2)
    logging.info(f"Saved feature importance to {importance_path}")
    
    return model, metrics, importance


def main():
    """Main function to parse arguments and train models"""
    parser = argparse.ArgumentParser(description="Train NBA prediction models for specific targets")
    
    parser.add_argument("--targets", type=str, nargs='+', 
                        default=['pts', 'reb', 'ast', 'fgm', 'fga', 'tptfgm', 'tptfga'],
                        help="Target variables to train models for (default: pts, reb, ast, fgm, fga, tptfgm, tptfga)")
    parser.add_argument("--ensemble", action="store_true", 
                        help="Enable ensemble stacking")
    parser.add_argument("--no-time-series", action="store_true",
                        help="Disable time series cross-validation")
    parser.add_argument("--gap", type=int, default=3,
                        help="Gap size for time series cross-validation (default: 3)")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Test size for train/test split (default: 0.2)")
    parser.add_argument("--process-data", action="store_true",
                        help="Run data processing before training")
                        
    # Feature engineering options
    feature_group = parser.add_argument_group('Feature Engineering Options')
    feature_group.add_argument("--optimize-features", action="store_true", default=True,
                        help="Enable feature optimization to remove redundant and low-importance features (default: on)")
    feature_group.add_argument("--no-optimize-features", action="store_false", dest="optimize_features",
                        help="Disable feature optimization")
    feature_group.add_argument("--validate-features", action="store_true", default=True,
                        help="Validate and fix derived features (default: on)")
    feature_group.add_argument("--no-validate-features", action="store_false", dest="validate_features",
                        help="Skip validation of derived features")
    feature_group.add_argument("--recache-features", action="store_true",
                        help="Force regeneration of feature cache if using processed data")
    
    args = parser.parse_args()
    
    # Process data if requested and available
    if args.process_data and data_pipeline_available:
        logging.info("Processing data...")
        processed_data = process_data(run_quality_checks=True)
        if processed_data is None:
            logging.error("Data processing failed")
            return
        
        logging.info("Engineering features...")
        engineered_data = engineer_all_features(
            processed_data,
            use_cache=not args.recache_features,
            remove_redundant_features=args.optimize_features,
            validate_derived_values=args.validate_features
        )
        if engineered_data is None:
            logging.error("Feature engineering failed")
            return
    
    # Train models for each target
    results = {}
    for target in args.targets:
        logging.info(f"\n{'='*50}\nTraining model for {target}\n{'='*50}")
        model, metrics, importance = train_specific_target(
            target,
            use_ensemble=args.ensemble,
            use_time_series=not args.no_time_series,
            gap=args.gap,
            test_size=args.test_size,
            remove_redundant_features=args.optimize_features,
            validate_features=args.validate_features
        )
        
        results[target] = {
            "model": model,
            "metrics": metrics,
            "importance": importance
        }
    
    # Print summary
    logging.info("\n\nTraining Results Summary:")
    logging.info("=" * 60)
    logging.info(f"{'Target':<12} {'R²':<10} {'MAE':<10} {'RMSE':<10}")
    logging.info("-" * 60)
    
    for target, result in results.items():
        metrics = result["metrics"]
        if metrics:
            # Handle the case where metrics are stored per target
            if target in metrics:
                target_metrics = metrics[target]
                r2 = target_metrics.get("r2", "N/A")
                mae = target_metrics.get("mae", "N/A")
                rmse = target_metrics.get("rmse", "N/A")
            else:
                # Try to get metrics directly
                r2 = metrics.get("r2", "N/A")
                mae = metrics.get("mae", "N/A")
                rmse = metrics.get("rmse", "N/A")
            
            if isinstance(r2, float): r2 = f"{r2:.4f}"
            if isinstance(mae, float): mae = f"{mae:.4f}"
            if isinstance(rmse, float): rmse = f"{rmse:.4f}"
            
            logging.info(f"{target:<12} {r2:<10} {mae:<10} {rmse:<10}")
    
    logging.info("=" * 60)


if __name__ == "__main__":
    main()