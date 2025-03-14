#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare different model types for the same target variable.
This script trains multiple model types on the same target and compares their performance.
"""

import os
import sys
import json
import logging
import argparse
import pandas as pd
from datetime import datetime

# Add the src directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.models.model_builder import train_model
from src.models.model_evaluator import evaluate_model, save_model_metrics, save_feature_importance
from src.visualization.model_visualizer import visualize_model_comparison

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(os.path.dirname(__file__), '../../logs/model_comparison.log'))
    ]
)
logger = logging.getLogger(__name__)

# Constants
MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models'))
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/processed'))
CURRENT_DATE = datetime.now().strftime('%Y%m%d')


def load_data(target_name):
    """
    Load training and testing data for a specific target
    
    Args:
        target_name (str): Name of the target variable (e.g., pts, reb, ast)
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test) data splits
    """
    try:
        # Load training data
        train_file = os.path.join(DATA_DIR, f"nba_{target_name}_train.csv")
        if not os.path.exists(train_file):
            logger.error(f"Training data file not found: {train_file}")
            return None, None, None, None
        
        train_data = pd.read_csv(train_file)
        
        # Load testing data
        test_file = os.path.join(DATA_DIR, f"nba_{target_name}_test.csv")
        if not os.path.exists(test_file):
            logger.error(f"Testing data file not found: {test_file}")
            return None, None, None, None
        
        test_data = pd.read_csv(test_file)
        
        # Split into features and target
        X_train = train_data.drop(target_name, axis=1)
        y_train = train_data[target_name]
        
        X_test = test_data.drop(target_name, axis=1)
        y_test = test_data[target_name]
        
        logger.info(f"Loaded data for target {target_name}: {X_train.shape[0]} training samples, {X_test.shape[0]} testing samples")
        return X_train, X_test, y_train, y_test
    
    except Exception as e:
        logger.error(f"Error loading data for {target_name}: {str(e)}")
        return None, None, None, None


def compare_models(target_name, model_types, tune_hyperparams=False, date_str=None):
    """
    Train and compare multiple model types for the same target variable
    
    Args:
        target_name (str): Name of the target variable (e.g., pts, reb, ast)
        model_types (list): List of model types to compare (e.g., ['random_forest', 'xgboost', 'lightgbm'])
        tune_hyperparams (bool): Whether to tune hyperparameters for each model
        date_str (str): Date string for file naming
        
    Returns:
        dict: Dictionary with model types as keys and metrics as values
    """
    if date_str is None:
        date_str = CURRENT_DATE
    
    # Load data
    X_train, X_test, y_train, y_test = load_data(target_name)
    if X_train is None:
        return None
    
    # Dictionary to store metrics for each model type
    comparison_results = {}
    
    # Train and evaluate each model type
    for model_type in model_types:
        logger.info(f"Training {model_type} model for target {target_name}")
        
        try:
            # Train the model
            model, train_time = train_model(
                X_train, y_train, 
                model_type=model_type,
                target_name=target_name,
                tune_hyperparams=tune_hyperparams
            )
            
            # Evaluate the model
            metrics = evaluate_model(model, X_test, y_test, target_name, visualize=False)
            metrics['train_time'] = train_time
            
            # Save model metrics
            metrics_file = save_model_metrics(metrics, target_name, model_type, date_str)
            
            # Save feature importance
            if hasattr(model, 'feature_importances_') or (hasattr(model, 'feature_importances_') and model_type in ['xgboost', 'lightgbm']):
                feature_importance = pd.DataFrame({
                    'feature': X_train.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                importance_file = save_feature_importance(
                    feature_importance, target_name, model_type, date_str
                )
            
            # Store metrics for comparison
            comparison_results[model_type] = metrics
            
            logger.info(f"Successfully trained and evaluated {model_type} model for {target_name}")
            logger.info(f"R² Score: {metrics['r2']:.4f}, MAE: {metrics['mae']:.4f}, RMSE: {metrics['rmse']:.4f}")
            
        except Exception as e:
            logger.error(f"Error training {model_type} model for {target_name}: {str(e)}")
    
    return comparison_results


def main():
    """Main function to compare model types"""
    parser = argparse.ArgumentParser(description='Compare different model types for NBA prediction')
    parser.add_argument('--target', type=str, required=True,
                        help='Target variable to predict (e.g., pts, reb, ast)')
    parser.add_argument('--models', type=str, nargs='+', 
                        default=['random_forest', 'xgboost', 'lightgbm'],
                        help='Model types to compare')
    parser.add_argument('--tune', action='store_true',
                        help='Tune hyperparameters for each model')
    parser.add_argument('--date', type=str, default=None,
                        help='Date string for file naming (format: YYYYMMDD)')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualization of model comparison')
    
    args = parser.parse_args()
    
    # Use current date if not provided
    date_str = args.date if args.date else CURRENT_DATE
    
    logger.info(f"Starting model comparison for target {args.target}")
    logger.info(f"Model types to compare: {', '.join(args.models)}")
    
    # Compare models
    comparison_results = compare_models(
        target_name=args.target,
        model_types=args.models,
        tune_hyperparams=args.tune,
        date_str=date_str
    )
    
    if comparison_results:
        # Create a target-specific directory for comparison results
        target_dir = os.path.join(MODELS_DIR, args.target)
        os.makedirs(target_dir, exist_ok=True)
        
        # Save comparison results
        comparison_file = os.path.join(
            target_dir, 
            f"model_comparison_{args.target}_{date_str}.json"
        )
        with open(comparison_file, 'w') as f:
            json.dump(comparison_results, f, indent=4)
        
        logger.info(f"Saved comparison results to {comparison_file}")
        
        # Generate visualization if requested
        if args.visualize:
            # Import here to avoid circular imports
            from src.visualization.model_visualizer import visualize_model_comparison
            
            # Format the comparison data for visualization
            viz_data = {args.target: {}}
            for model_type, metrics in comparison_results.items():
                viz_data[f"{args.target}_{model_type}"] = metrics
            
            # Generate visualization
            visualize_model_comparison(
                comparison_data=viz_data,
                model_type=None,  # No specific model type since we're comparing different types
                date_str=date_str,
                save_fig=True,
                show_fig=True
            )
            
            logger.info("Generated model comparison visualization")
    
    logger.info("Model comparison completed")


if __name__ == "__main__":
    main()
