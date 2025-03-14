#!/usr/bin/env python3
"""
Model builder module for NBA prediction models

This module contains functions for building, training, and evaluating models
"""

import os
import logging
import pandas as pd
import numpy as np
import joblib
import glob
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

# Try to import optional libraries
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("XGBoost not available. Install with 'pip install xgboost' to use XGBoost models.")

try:
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logging.warning("LightGBM not available. Install with 'pip install lightgbm' to use LightGBM models.")

try:
    from skopt import BayesSearchCV
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False
    logging.warning("scikit-optimize not available. Install with 'pip install scikit-optimize' to use Bayesian optimization.")

# Import from src modules
try:
    from src.utils.config import DATA_DIR, MODELS_DIR, CURRENT_DATE
    # Try to import the model evaluator module
    try:
        from src.models.model_evaluator import evaluate_model as evaluator_evaluate_model
        from src.models.model_evaluator import save_model_metrics, save_feature_importance
        from src.visualization.model_visualizer import (
            visualize_feature_importance,
            visualize_model_metrics,
            visualize_prediction_analysis
        )
        EVALUATOR_AVAILABLE = True
    except ImportError:
        EVALUATOR_AVAILABLE = False
        logging.warning("Model evaluator module not available. Using built-in evaluation functions.")
except ImportError:
    # Try direct import for when running from models directory
    try:
        from utils.config import DATA_DIR, MODELS_DIR, CURRENT_DATE
        # Try to import the model evaluator module
        try:
            from model_evaluator import evaluate_model as evaluator_evaluate_model
            from model_evaluator import save_model_metrics, save_feature_importance
            from visualization.model_visualizer import (
                visualize_feature_importance,
                visualize_model_metrics,
                visualize_prediction_analysis
            )
            EVALUATOR_AVAILABLE = True
        except ImportError:
            EVALUATOR_AVAILABLE = False
            logging.warning("Model evaluator module not available. Using built-in evaluation functions.")
    except ImportError:
        logging.error("Failed to import config module. Make sure you're running from the project root.")
        # Use default values
        DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")
        MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models")
        CURRENT_DATE = datetime.now().strftime("%Y%m%d")
        EVALUATOR_AVAILABLE = False

def load_training_data(data_path=None):
    """
    Load training data for model building
    
    Args:
        data_path (str, optional): Path to the data file. If None, will try to find the latest engineered data file.
        
    Returns:
        pandas.DataFrame: The loaded data
    """
    if data_path is None:
        # Try to find the latest engineered data file
        try:
            # Check if the engineered directory exists
            engineered_dir = os.path.join(DATA_DIR, "engineered")
            if not os.path.exists(engineered_dir):
                logging.warning(f"Engineered data directory not found: {engineered_dir}")
                logging.info("Trying to find data in the processed directory...")
                
                # Try the processed directory instead
                processed_dir = os.path.join(DATA_DIR, "processed")
                if not os.path.exists(processed_dir):
                    logging.error(f"Processed data directory not found: {processed_dir}")
                    return None
                
                # Find the latest processed data file
                processed_files = glob.glob(os.path.join(processed_dir, "processed_nba_data_*.csv"))
                if not processed_files:
                    logging.error("No processed data files found")
                    return None
                
                # Sort by modification time (newest first)
                processed_files.sort(key=os.path.getmtime, reverse=True)
                data_path = processed_files[0]
                logging.info(f"Using latest processed data file: {data_path}")
            else:
                # Find the latest engineered data file
                engineered_files = glob.glob(os.path.join(engineered_dir, "engineered_nba_data_*.csv"))
                if not engineered_files:
                    logging.error("No engineered data files found")
                    return None
                
                # Sort by modification time (newest first)
                engineered_files.sort(key=os.path.getmtime, reverse=True)
                data_path = engineered_files[0]
                logging.info(f"Using latest engineered data file: {data_path}")
        except Exception as e:
            logging.error(f"Error finding latest data file: {str(e)}")
            return None
    
    # Load the data
    try:
        logging.info(f"Loading data from {data_path}")
        data = pd.read_csv(data_path)
        logging.info(f"Loaded data with {data.shape[0]} rows and {data.shape[1]} columns")
        return data
    except Exception as e:
        logging.error(f"Error loading data: {str(e)}")
        return None

def create_feature_matrix(data, target_name, min_minutes=5, test_size=0.2, random_state=42):
    """
    Create feature matrix and target vector for model training
    
    Args:
        data (pandas.DataFrame): The input data
        target_name (str): Name of the target variable
        min_minutes (int): Minimum minutes played to include in training
        test_size (float): Fraction of data to use for testing
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: X (features), y (target), feature_names (list of feature names)
    """
    # Filter out rows with too few minutes played
    if 'minutes' in data.columns:
        filtered_data = data[data['minutes'] >= min_minutes].copy()
        logging.info(f"Filtered data to {filtered_data.shape[0]} rows with at least {min_minutes} minutes played")
    else:
        filtered_data = data.copy()
        logging.warning("No 'minutes' column found. Using all data rows.")
    
    # Check if target column exists
    if target_name not in filtered_data.columns:
        logging.error(f"Target column '{target_name}' not found in data")
        return None, None, None
    
    # Define features to exclude
    exclude_columns = [
        # Target variables
        'pts', 'reb', 'ast', 'fgm', 'fga', 'tptfgm', 'tptfga',
        # Identifiers and dates
        'player_id', 'player_name', 'team_id', 'team_name', 'game_id', 'game_date',
        # Future information
        'next_game_date', 'next_game_id', 'next_team_id', 'next_team_name',
        # Columns with too many missing values
        'fantasy_points', 'fantasy_points_fanduel', 'fantasy_points_draftkings',
        # Columns that might cause data leakage
        'minutes', 'seconds_played',
        # Other columns to exclude
        'season', 'season_type', 'is_home', 'is_away', 'is_win', 'is_loss',
        # Date components
        'year', 'month', 'day', 'dayofweek', 'dayofyear', 'quarter', 'week',
        # Derived target variables
        f'{target_name}_rolling_avg', f'{target_name}_rolling_std', f'{target_name}_rolling_min', 
        f'{target_name}_rolling_max', f'{target_name}_rolling_median',
        f'{target_name}_vs_team_avg', f'{target_name}_home_avg', f'{target_name}_away_avg',
        f'{target_name}_last_5', f'{target_name}_last_10', f'{target_name}_last_15',
        f'{target_name}_season_avg', f'{target_name}_last_3_avg', f'{target_name}_last_5_avg',
        # Add any other columns that should be excluded
    ]
    
    # Only exclude columns that actually exist in the data
    exclude_columns = [col for col in exclude_columns if col in filtered_data.columns and col != target_name]
    
    # Create feature matrix
    feature_columns = [col for col in filtered_data.columns if col not in exclude_columns and col != target_name]
    
    # Identify and exclude non-numeric columns
    non_numeric_columns = []
    for col in feature_columns:
        try:
            pd.to_numeric(filtered_data[col])
        except (ValueError, TypeError):
            non_numeric_columns.append(col)
            logging.warning(f"Excluding non-numeric column: {col}")
    
    # Remove non-numeric columns from feature list
    feature_columns = [col for col in feature_columns if col not in non_numeric_columns]
    
    # Log the number of features
    logging.info(f"Using {len(feature_columns)} features for model training")
    
    # Create feature matrix and target vector
    X = filtered_data[feature_columns].copy()
    y = filtered_data[target_name].copy()
    
    # Handle missing values
    X = X.replace([np.inf, -np.inf], np.nan)
    
    # Log the number of missing values
    missing_values = X.isna().sum().sum()
    if missing_values > 0:
        logging.warning(f"Found {missing_values} missing values in the feature matrix")
        logging.info("Missing values will be imputed during model training")
    
    return X, y, feature_columns

def train_model(X, y, model_type='random_forest', target_name=None, tune_hyperparams=False):
    """
    Train a model on the given data
    
    Args:
        X (pandas.DataFrame): Feature matrix
        y (pandas.Series): Target vector
        model_type (str): Type of model to train
        target_name (str, optional): Name of the target variable
        tune_hyperparams (bool): Whether to tune hyperparameters
        
    Returns:
        tuple: (trained_model, training_time_seconds)
    """
    logging.info(f"Training {model_type} model...")
    
    # Record start time
    start_time = datetime.now()
    
    # Create the model based on the specified type
    if model_type == 'random_forest':
        if tune_hyperparams and SKOPT_AVAILABLE:
            logging.info("Tuning hyperparameters for Random Forest...")
            model = BayesSearchCV(
                RandomForestRegressor(random_state=42),
                {
                    'n_estimators': (50, 300),
                    'max_depth': (5, 30),
                    'min_samples_split': (2, 20),
                    'min_samples_leaf': (1, 10)
                },
                n_iter=20,
                cv=5,
                n_jobs=-1,
                random_state=42
            )
        else:
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42,
                n_jobs=-1
            )
    elif model_type == 'xgboost':
        if not XGBOOST_AVAILABLE:
            logging.error("XGBoost not available. Install with 'pip install xgboost'")
            return None, 0
        
        if tune_hyperparams and SKOPT_AVAILABLE:
            logging.info("Tuning hyperparameters for XGBoost...")
            model = BayesSearchCV(
                XGBRegressor(random_state=42),
                {
                    'n_estimators': (50, 300),
                    'max_depth': (3, 10),
                    'learning_rate': (0.01, 0.3, 'log-uniform'),
                    'subsample': (0.6, 1.0),
                    'colsample_bytree': (0.6, 1.0)
                },
                n_iter=20,
                cv=5,
                n_jobs=-1,
                random_state=42
            )
        else:
            model = XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            )
    elif model_type == 'lightgbm':
        if not LIGHTGBM_AVAILABLE:
            logging.error("LightGBM not available. Install with 'pip install lightgbm'")
            return None, 0
        
        if tune_hyperparams and SKOPT_AVAILABLE:
            logging.info("Tuning hyperparameters for LightGBM...")
            model = BayesSearchCV(
                LGBMRegressor(random_state=42),
                {
                    'n_estimators': (50, 300),
                    'max_depth': (3, 10),
                    'learning_rate': (0.01, 0.3, 'log-uniform'),
                    'subsample': (0.6, 1.0),
                    'colsample_bytree': (0.6, 1.0)
                },
                n_iter=20,
                cv=5,
                n_jobs=-1,
                random_state=42
            )
        else:
            model = LGBMRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            )
    else:
        logging.error(f"Unsupported model type: {model_type}")
        return None, 0
    
    # Train the model
    try:
        model.fit(X, y)
        
        # If we used BayesSearchCV, extract the best estimator
        if tune_hyperparams and SKOPT_AVAILABLE:
            logging.info(f"Best hyperparameters: {model.best_params_}")
            model = model.best_estimator_
        
        # Calculate training time
        end_time = datetime.now()
        train_time = (end_time - start_time).total_seconds()
        
        logging.info(f"Model training completed in {train_time:.2f} seconds")
        
        return model, train_time
    except Exception as e:
        logging.error(f"Error training model: {str(e)}")
        return None, 0

def evaluate_model(model, X, y, target_name, use_cv=True, cv_folds=5, use_time_series_cv=False):
    """
    Evaluate a trained model
    
    Args:
        model (object): The trained model
        X (pandas.DataFrame): Feature matrix
        y (pandas.Series): Target vector
        target_name (str): Name of the target variable
        use_cv (bool): Whether to use cross-validation
        cv_folds (int): Number of cross-validation folds
        use_time_series_cv (bool): Whether to use time series cross-validation
        
    Returns:
        dict: Evaluation metrics
    """
    if EVALUATOR_AVAILABLE:
        return evaluator_evaluate_model(model, X, y, target_name, use_cv, cv_folds, use_time_series_cv)
    else:
        # Make predictions
        try:
            y_pred = model.predict(X)
        except Exception as e:
            logging.error(f"Error making predictions: {str(e)}")
            return None
        
        # Calculate metrics
        metrics = {}
        metrics['r2'] = r2_score(y, y_pred)
        metrics['mae'] = mean_absolute_error(y, y_pred)
        metrics['rmse'] = np.sqrt(mean_squared_error(y, y_pred))
        
        logging.info(f"Model evaluation for {target_name}:")
        logging.info(f"  R² score: {metrics['r2']:.4f}")
        logging.info(f"  Mean absolute error: {metrics['mae']:.4f}")
        logging.info(f"  Root mean squared error: {metrics['rmse']:.4f}")
        
        # Cross-validation if requested
        if use_cv:
            try:
                if use_time_series_cv:
                    cv = TimeSeriesSplit(n_splits=cv_folds)
                else:
                    cv = cv_folds
                    
                cv_scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
                metrics['cv_r2_mean'] = cv_scores.mean()
                metrics['cv_r2_std'] = cv_scores.std()
                
                logging.info(f"  Cross-validation R² score: {metrics['cv_r2_mean']:.4f} ± {metrics['cv_r2_std']:.4f}")
            except Exception as e:
                logging.error(f"Error during cross-validation: {str(e)}")
        
        return metrics

def analyze_feature_importance(model, X, y, feature_names, n_top_features=20):
    """
    Analyze feature importance from a trained model
    
    Args:
        model (object): The trained model
        X (pandas.DataFrame): Feature matrix
        y (pandas.Series): Target vector
        feature_names (list): List of feature names
        n_top_features (int): Number of top features to return
        
    Returns:
        dict: Feature importance scores
    """
    # Extract the underlying model if it's wrapped in a search/CV object
    if hasattr(model, 'best_estimator_'):
        model = model.best_estimator_
    
    # Check if model has feature_importances_
    if not hasattr(model, 'feature_importances_'):
        # Try to get the model from the pipeline
        if hasattr(model, 'named_steps') and 'model' in model.named_steps:
            model = model.named_steps['model']
            if not hasattr(model, 'feature_importances_'):
                logging.warning("Model does not have feature_importances_ attribute")
                return None
        else:
            # For models without direct feature_importances_, try to use permutation importance
            try:
                from sklearn.inspection import permutation_importance
                logging.info("Computing permutation feature importance...")
                perm_importance = permutation_importance(model, X, y, n_repeats=10, random_state=42)
                importances = perm_importance.importances_mean
                
                # Create a dictionary of feature importances
                importance_dict = {name: float(importance) for name, importance in zip(feature_names, importances)}
                
                # Sort by importance (descending)
                importance_dict = {k: v for k, v in sorted(importance_dict.items(), key=lambda item: item[1], reverse=True)}
                
                # Log top features
                logging.info(f"Top {n_top_features} features by permutation importance:")
                for i, (feature, importance) in enumerate(list(importance_dict.items())[:n_top_features]):
                    logging.info(f"  {feature}: {importance:.4f}")
                
                return importance_dict
            except Exception as e:
                logging.warning(f"Could not compute permutation importance: {str(e)}")
                return None
    
    # Get feature importances
    importances = model.feature_importances_
    
    # Create a dictionary of feature importances
    importance_dict = {name: float(importance) for name, importance in zip(feature_names, importances)}
    
    # Sort by importance (descending)
    importance_dict = {k: v for k, v in sorted(importance_dict.items(), key=lambda item: item[1], reverse=True)}
    
    # Log top features
    logging.info(f"Top {n_top_features} features by importance:")
    for i, (feature, importance) in enumerate(list(importance_dict.items())[:n_top_features]):
        logging.info(f"  {feature}: {importance:.4f}")
    
    return importance_dict

def save_model(model, model_path):
    """
    Save a trained model to disk
    
    Args:
        model (object): The trained model
        model_path (str): Path to save the model
        
    Returns:
        bool: Whether the save was successful
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save the model
        joblib.dump(model, model_path)
        return True
    except Exception as e:
        logging.error(f"Error saving model: {str(e)}")
        return False

def save_metrics(metrics, target_name, model_type, date_str=None):
    """
    Save model metrics to a JSON file
    
    Args:
        metrics (dict): Model metrics
        target_name (str): Name of the target variable
        model_type (str): Type of model
        date_str (str): Date string for the filename
        
    Returns:
        str: Path to the saved metrics file
    """
    if date_str is None:
        date_str = CURRENT_DATE
    
    # Create metrics directory if it doesn't exist
    metrics_dir = os.path.join(MODELS_DIR, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Create a filename with a professional format
    filename = f"{target_name}_{model_type}_metrics_{date_str}.json"
    metrics_path = os.path.join(metrics_dir, filename)
    
    # Save metrics to JSON
    try:
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        logging.info(f"Saved metrics to {metrics_path}")
        
        # If the evaluator module is available, use it to generate visualizations
        if EVALUATOR_AVAILABLE:
            try:
                # Save metrics using the evaluator module
                save_model_metrics(metrics, target_name, model_type, date_str)
                
                # Generate visualizations
                visualize_model_metrics(
                    target_name=target_name,
                    model_type=model_type,
                    date_str=date_str,
                    save_fig=True,
                    show_fig=False
                )
            except Exception as e:
                logging.error(f"Error generating metrics visualization: {str(e)}")
        
        return metrics_path
    except Exception as e:
        logging.error(f"Error saving metrics: {str(e)}")
        return None

def save_feature_importance(importance, target_name, model_type, date_str=None):
    """
    Save feature importance to a JSON file
    
    Args:
        importance (dict): Feature importance
        target_name (str): Name of the target variable
        model_type (str): Type of model
        date_str (str): Date string for the filename
        
    Returns:
        str: Path to the saved feature importance file
    """
    if date_str is None:
        date_str = CURRENT_DATE
    
    # Create feature importance directory if it doesn't exist
    importance_dir = os.path.join(MODELS_DIR, "feature_importance")
    os.makedirs(importance_dir, exist_ok=True)
    
    # Create a filename with a professional format
    filename = f"{target_name}_{model_type}_feature_importance_{date_str}.json"
    importance_path = os.path.join(importance_dir, filename)
    
    # Save feature importance to JSON
    try:
        with open(importance_path, 'w') as f:
            json.dump(importance, f, indent=4)
        logging.info(f"Saved feature importance to {importance_path}")
        
        # If the evaluator module is available, use it to save feature importance and generate visualizations
        if EVALUATOR_AVAILABLE:
            try:
                # Save feature importance using the evaluator module
                save_feature_importance(importance, target_name, model_type, date_str)
                
                # Generate visualizations
                visualize_feature_importance(
                    target_name=target_name,
                    model_type=model_type,
                    date_str=date_str,
                    top_n=20,
                    save_fig=True,
                    show_fig=False
                )
            except Exception as e:
                logging.error(f"Error generating feature importance visualization: {str(e)}")
        
        return importance_path
    except Exception as e:
        logging.error(f"Error saving feature importance: {str(e)}")
        return None

def load_model(model_path):
    """
    Load a trained model from disk
    
    Args:
        model_path (str): Path to the saved model
        
    Returns:
        object: The loaded model
    """
    try:
        # Check if file exists
        if not os.path.exists(model_path):
            logging.error(f"Model file not found: {model_path}")
            return None
        
        # Load the model
        model = joblib.load(model_path)
        return model
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        return None
