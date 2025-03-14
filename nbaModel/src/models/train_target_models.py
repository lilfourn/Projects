#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module for training NBA prediction models for specific targets
"""

import os
import sys
import logging
import argparse
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Import from src modules
try:
    from src.utils.config import DATA_DIR, MODELS_DIR, CURRENT_DATE
    from src.models.model_builder import train_model, analyze_feature_importance
    from src.models.model_evaluator import (
        evaluate_model, save_model_metrics, save_feature_importance, get_target_model_dir
    )
    from src.visualization.model_visualizer import (
        visualize_feature_importance, visualize_model_metrics, visualize_prediction_analysis
    )
except ImportError:
    # Try direct import for when running from models directory
    try:
        from utils.config import DATA_DIR, MODELS_DIR, CURRENT_DATE
        from model_builder import train_model, analyze_feature_importance
        from model_evaluator import (
            evaluate_model, save_model_metrics, save_feature_importance, get_target_model_dir
        )
        from visualization.model_visualizer import (
            visualize_feature_importance, visualize_model_metrics, visualize_prediction_analysis
        )
    except ImportError:
        logging.error("Failed to import required modules. Make sure you're running from the project root.")
        # Use default values
        DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")
        MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models")
        CURRENT_DATE = datetime.now().strftime("%Y%m%d")
        
        # Import functions directly from the local files
        import sys
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from models.model_builder import train_model, analyze_feature_importance
        from models.model_evaluator import (
            evaluate_model, save_model_metrics, save_feature_importance, get_target_model_dir
        )
        from visualization.model_visualizer import (
            visualize_feature_importance, visualize_model_metrics, visualize_prediction_analysis
        )

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train NBA prediction models for specific targets')
    
    parser.add_argument('--targets', type=str, nargs='+', required=True,
                        help='Target variables to predict (e.g., pts reb ast)')
    parser.add_argument('--model-type', type=str, default='random_forest',
                        choices=['random_forest', 'xgboost', 'lightgbm'],
                        help='Type of model to train')
    parser.add_argument('--tune-hyperparams', action='store_true',
                        help='Tune hyperparameters for the model')
    parser.add_argument('--generate-visualizations', action='store_true',
                        help='Generate visualizations for model evaluation')
    parser.add_argument('--date', type=str, default=None,
                        help='Date string for file naming (format: YYYYMMDD)')
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save the trained model')
    
    return parser.parse_args()

def train_target_model(target_name, model_type='random_forest', tune_hyperparams=False, 
                      visualize=True, save_model=True, date_str=None):
    """
    Train a model for a specific target variable
    
    Args:
        target_name (str): Name of the target variable (e.g., pts, reb, ast)
        model_type (str): Type of model to train (random_forest, xgboost, lightgbm)
        tune_hyperparams (bool): Whether to tune hyperparameters
        visualize (bool): Whether to generate visualizations
        save_model (bool): Whether to save the trained model
        date_str (str): Date string for file naming
        
    Returns:
        tuple: Trained model and evaluation metrics
    """
    if date_str is None:
        date_str = CURRENT_DATE
        
    logging.info(f"Training {model_type} model for target: {target_name}")
    
    # Load training data
    train_file = os.path.join(DATA_DIR, "processed", f"nba_{target_name}_train.csv")
    if not os.path.exists(train_file):
        logging.error(f"Training data file not found: {train_file}")
        return None, None
    
    train_data = pd.read_csv(train_file)
    
    # Load testing data
    test_file = os.path.join(DATA_DIR, "processed", f"nba_{target_name}_test.csv")
    if not os.path.exists(test_file):
        logging.error(f"Testing data file not found: {test_file}")
        return None, None
    
    test_data = pd.read_csv(test_file)
    
    # Split into features and target
    X_train = train_data.drop(target_name, axis=1)
    y_train = train_data[target_name]
    
    X_test = test_data.drop(target_name, axis=1)
    y_test = test_data[target_name]
    
    # Train the model
    model, train_time = train_model(
        X_train, y_train, 
        model_type=model_type,
        target_name=target_name,
        tune_hyperparams=tune_hyperparams
    )
    
    # Evaluate the model
    metrics = evaluate_model(model, X_test, y_test, target_name, visualize=visualize)
    metrics['train_time'] = train_time
    
    # Save model metrics
    metrics_file = save_model_metrics(metrics, target_name, model_type, date_str)
    
    # Save feature importance using the analyze_feature_importance function
    feature_importance_dict = analyze_feature_importance(
        model, X_train, y_train, X_train.columns.tolist(), n_top_features=20
    )
    
    if feature_importance_dict:
        feature_importance = pd.DataFrame({
            'feature': list(feature_importance_dict.keys()),
            'importance': list(feature_importance_dict.values())
        })
        
        importance_file = save_feature_importance(
            feature_importance, target_name, model_type, date_str
        )
    
    # Save the model
    if save_model:
        # Get target-specific directory
        target_dir = get_target_model_dir(target_name)
        
        # Create a professional filename
        filename = f"nba_{target_name}_{model_type}_model_{date_str}.joblib"
        model_file = os.path.join(target_dir, filename)
        
        try:
            joblib.dump(model, model_file)
            logging.info(f"Saved model to {model_file}")
        except Exception as e:
            logging.error(f"Error saving model: {str(e)}")
    
    # Generate visualizations if requested
    if visualize:
        try:
            # Visualize feature importance if we have it
            if feature_importance_dict:
                visualize_feature_importance(
                    target_name=target_name,
                    date_str=date_str,
                    top_n=20,
                    save_fig=True,
                    show_fig=False
                )
            
            # Visualize model metrics
            visualize_model_metrics(
                target_name=target_name,
                date_str=date_str,
                save_fig=True,
                show_fig=False
            )
            
            # Visualize prediction analysis
            y_pred = model.predict(X_test)
            visualize_prediction_analysis(
                target_name=target_name,
                actual=y_test,
                predicted=y_pred,
                date_str=date_str,
                save_fig=True,
                show_fig=False
            )
        except Exception as e:
            logging.error(f"Error generating visualizations: {str(e)}")
    
    return model, metrics

def main():
    """Main function to train models for specific targets"""
    args = parse_args()
    
    results = {}
    
    for target in args.targets:
        logging.info(f"Training model for target: {target}")
        
        model, metrics = train_target_model(
            target,
            model_type=args.model_type,
            tune_hyperparams=args.tune_hyperparams,
            visualize=args.generate_visualizations,
            save_model=not args.no_save,
            date_str=args.date
        )
        
        if model is None:
            logging.error(f"Failed to train model for target: {target}")
            continue
        
        results[target] = {
            "model": model,
            "metrics": metrics
        }
    
    # Print summary
    logging.info("\n\nTraining Results Summary:")
    logging.info("=" * 60)
    logging.info(f"{'Target':<12} {'R²':<10} {'MAE':<10} {'RMSE':<10}")
    logging.info("-" * 60)
    
    for target, result in results.items():
        metrics = result["metrics"]
        if metrics:
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
