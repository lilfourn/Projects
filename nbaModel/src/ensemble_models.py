"""
Model ensembling capabilities for NBA prediction models
"""

import numpy as np
import pandas as pd
import os
import glob
import joblib
import json
import logging
from datetime import datetime
from sklearn.ensemble import VotingRegressor, StackingRegressor
from sklearn.linear_model import Ridge
import xgboost as xgb

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_model(model_path):
    """
    Load a model from file
    
    Args:
        model_path (str): Path to model file
        
    Returns:
        object: Loaded model
    """
    try:
        model = joblib.load(model_path)
        logging.info(f"Loaded model from {model_path}")
        return model
    except Exception as e:
        logging.error(f"Error loading model {model_path}: {str(e)}")
        return None

def find_models_by_target(target, models_dir=None):
    """
    Find all models for a specific target
    
    Args:
        target (str): Target name (e.g., 'pts', 'ast')
        models_dir (str, optional): Directory to search for models
        
    Returns:
        list: List of model paths
    """
    if models_dir is None:
        models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
    
    # Search for model files with target name and model type pattern
    # Look specifically for our new consolidated model structure
    model_files = glob.glob(os.path.join(models_dir, f"nba_{target}_*_*.joblib"))
    
    # If no results, try more general search
    if not model_files:
        model_files = glob.glob(os.path.join(models_dir, f"nba_{target}_*.joblib"))
    
    # Sort by date (newest first)
    model_files = sorted(model_files, reverse=True)
    
    return model_files

def load_models_by_type(target, models_dir=None, model_types=None):
    """
    Load models of specific types for a target
    
    Args:
        target (str): Target name
        models_dir (str, optional): Directory to search for models
        model_types (list, optional): List of model types to load
                                     (e.g., ['random_forest', 'xgboost'])
        
    Returns:
        dict: Dictionary of models by type
    """
    if models_dir is None:
        models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
    
    if model_types is None:
        model_types = ['random_forest', 'xgboost', 'gradient_boosting']
    
    # Find all model files for this target
    model_files = find_models_by_target(target, models_dir)
    
    # Load models by type
    loaded_models = {}
    
    for model_file in model_files:
        # Try to determine model type from filename
        filename = os.path.basename(model_file)
        detected_type = None
        
        # Check for model type in filename
        for model_type in model_types:
            if model_type.lower() in filename.lower():
                detected_type = model_type
                break
        
        # If no type detected in filename, try to load and detect from the model
        if detected_type is None:
            try:
                model = load_model(model_file)
                
                # Check model type
                if hasattr(model, 'get_booster'):
                    detected_type = 'xgboost'
                elif hasattr(model, 'estimators_') and hasattr(model, 'n_estimators'):
                    # Check if it's random forest or gradient boosting
                    if hasattr(model, 'bootstrap') and model.bootstrap:
                        detected_type = 'random_forest'
                    else:
                        detected_type = 'gradient_boosting'
                elif hasattr(model, 'estimators'):
                    detected_type = 'ensemble'
                
                # Skip if we couldn't detect or it's not one of the requested types
                if detected_type is None or detected_type not in model_types:
                    continue
                
                # Store the model
                if detected_type not in loaded_models:
                    loaded_models[detected_type] = model
                    logging.info(f"Loaded {detected_type} model for {target} from {model_file}")
            except:
                continue
        else:
            # Load the model if we detected the type from the filename
            if detected_type not in loaded_models:
                model = load_model(model_file)
                if model is not None:
                    loaded_models[detected_type] = model
                    logging.info(f"Loaded {detected_type} model for {target} from {model_file}")
    
    return loaded_models

def create_voting_ensemble(models, weights=None):
    """
    Create a voting ensemble from multiple models
    
    Args:
        models (dict): Dictionary of models by name
        weights (list, optional): List of weights for each model
        
    Returns:
        object: Voting ensemble model
    """
    if not models:
        logging.error("No models provided for ensemble")
        return None
    
    # Create list of (name, model) tuples for VotingRegressor
    estimators = [(name, model) for name, model in models.items()]
    
    if not estimators:
        logging.error("No valid estimators for ensemble")
        return None
    
    # Create voting ensemble
    # Note: VotingRegressor for regression doesn't have a 'voting' parameter
    ensemble = VotingRegressor(estimators=estimators, weights=weights)
    
    logging.info(f"Created voting ensemble with {len(estimators)} models")
    return ensemble

def create_stacking_ensemble(models, meta_model=None, cv=5):
    """
    Create a stacking ensemble from multiple models
    
    Args:
        models (dict): Dictionary of models by name
        meta_model (object, optional): Meta-model for stacking
        cv (int): Number of cross-validation folds
        
    Returns:
        object: Stacking ensemble model
    """
    if not models:
        logging.error("No models provided for ensemble")
        return None
    
    # Create list of (name, model) tuples for StackingRegressor
    estimators = [(name, model) for name, model in models.items()]
    
    if not estimators:
        logging.error("No valid estimators for ensemble")
        return None
    
    # Default to Ridge regression for meta-model
    if meta_model is None:
        meta_model = Ridge()
    
    # Create stacking ensemble
    ensemble = StackingRegressor(estimators=estimators, final_estimator=meta_model, cv=cv)
    
    logging.info(f"Created stacking ensemble with {len(estimators)} models and {meta_model.__class__.__name__} meta-model")
    return ensemble

def get_best_weight_scheme(models, X_val, y_val):
    """
    Determine the best weight scheme for ensemble based on validation performance
    
    Args:
        models (dict): Dictionary of models by name
        X_val (pd.DataFrame): Validation features
        y_val (pd.Series): Validation target
        
    Returns:
        list: Optimal weights for each model
    """
    if not models or X_val is None or y_val is None:
        # No validation data, use equal weights
        return None
    
    # Get predictions from each model
    predictions = {}
    for name, model in models.items():
        try:
            predictions[name] = model.predict(X_val)
        except:
            logging.warning(f"Could not get predictions from {name} model")
    
    if not predictions:
        return None
    
    # Calculate MSE for each model
    mse = {}
    for name, preds in predictions.items():
        mse[name] = np.mean((preds - y_val) ** 2)
    
    # Convert MSE to weights (lower MSE = higher weight)
    # Use inverse of MSE
    inverse_mse = {name: 1/score for name, score in mse.items()}
    
    # Normalize weights to sum to 1
    total = sum(inverse_mse.values())
    weights = [inverse_mse[name]/total for name in models.keys()]
    
    logging.info(f"Determined optimal weights for ensemble: {list(zip(models.keys(), weights))}")
    return weights

def train_ensemble_models(target, models_dir=None, output_dir=None, ensemble_types=None,
                          model_types=None, test_data=None):
    """
    Train ensemble models for a specific target
    
    Args:
        target (str): Target name
        models_dir (str, optional): Directory with base models
        output_dir (str, optional): Directory to save ensemble models
        ensemble_types (list, optional): Types of ensembles to create
                                        (e.g., ['voting', 'stacking'])
        model_types (list, optional): Types of models to include in ensembles
        test_data (tuple, optional): (X_test, y_test) for evaluation
        
    Returns:
        dict: Dictionary of ensemble models
    """
    if models_dir is None:
        models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
    
    if output_dir is None:
        output_dir = models_dir
    
    if ensemble_types is None:
        ensemble_types = ['voting', 'stacking']
    
    if model_types is None:
        model_types = ['random_forest', 'xgboost', 'gradient_boosting']
    
    # Load base models
    base_models = load_models_by_type(target, models_dir, model_types)
    
    if len(base_models) < 2:
        logging.error(f"Need at least 2 models for ensemble, only found {len(base_models)}")
        return {}
    
    # Create ensembles
    ensembles = {}
    
    # Today's date for filenames
    date_suffix = datetime.now().strftime("%Y%m%d")
    
    # Determine weights if we have test data
    weights = None
    if test_data is not None:
        X_test, y_test = test_data
        weights = get_best_weight_scheme(base_models, X_test, y_test)
    
    # Create voting ensemble
    if 'voting' in ensemble_types:
        voting_model = create_voting_ensemble(base_models, weights=weights)
        if voting_model is not None:
            ensembles['voting'] = voting_model
            
            # Save the model
            model_path = os.path.join(output_dir, f"nba_{target}_voting_ensemble_{date_suffix}.joblib")
            try:
                joblib.dump(voting_model, model_path)
                logging.info(f"Saved voting ensemble to {model_path}")
            except Exception as e:
                logging.error(f"Error saving voting ensemble: {str(e)}")
    
    # Create stacking ensemble
    if 'stacking' in ensemble_types:
        # Use XGBoost as meta-model if available, otherwise Ridge
        meta_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, subsample=0.8, 
                                      colsample_bytree=0.8, max_depth=3)
        
        stacking_model = create_stacking_ensemble(base_models, meta_model=meta_model, cv=5)
        if stacking_model is not None:
            ensembles['stacking'] = stacking_model
            
            # Save the model
            model_path = os.path.join(output_dir, f"nba_{target}_stacking_ensemble_{date_suffix}.joblib")
            
            # If test data available, train the stacking model
            if test_data is not None:
                X_test, y_test = test_data
                try:
                    stacking_model.fit(X_test, y_test)
                    logging.info(f"Trained stacking ensemble on {len(X_test)} samples")
                except Exception as e:
                    logging.error(f"Error training stacking ensemble: {str(e)}")
            
            # Save model
            try:
                joblib.dump(stacking_model, model_path)
                logging.info(f"Saved stacking ensemble to {model_path}")
            except Exception as e:
                logging.error(f"Error saving stacking ensemble: {str(e)}")
    
    return ensembles

def evaluate_ensemble(ensemble, X_test, y_test, base_models=None):
    """
    Evaluate ensemble performance and compare to base models
    
    Args:
        ensemble: Ensemble model
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test target
        base_models (dict, optional): Dictionary of base models for comparison
        
    Returns:
        dict: Evaluation metrics
    """
    if X_test is None or y_test is None:
        logging.error("No test data provided for evaluation")
        return {}
    
    # Calculate predictions
    try:
        y_pred = ensemble.predict(X_test)
    except Exception as e:
        logging.error(f"Error making predictions with ensemble: {str(e)}")
        return {}
    
    # Calculate metrics
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    metrics = {
        'mse': mean_squared_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'mae': mean_absolute_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred)
    }
    
    # Calculate base model metrics for comparison
    if base_models:
        base_metrics = {}
        
        for name, model in base_models.items():
            try:
                base_pred = model.predict(X_test)
                base_metrics[name] = {
                    'mse': mean_squared_error(y_test, base_pred),
                    'rmse': np.sqrt(mean_squared_error(y_test, base_pred)),
                    'mae': mean_absolute_error(y_test, base_pred),
                    'r2': r2_score(y_test, base_pred)
                }
            except:
                logging.warning(f"Could not evaluate {name} model")
        
        metrics['base_models'] = base_metrics
        
        # Calculate improvement over best base model
        if base_metrics:
            best_rmse = min(model['rmse'] for model in base_metrics.values())
            improvement = (best_rmse - metrics['rmse']) / best_rmse * 100
            metrics['improvement'] = improvement
            
            logging.info(f"Ensemble RMSE: {metrics['rmse']:.4f}, best base model RMSE: {best_rmse:.4f}")
            logging.info(f"Improvement: {improvement:.2f}%")
    
    return metrics

def create_ensemble_for_all_targets(models_dir=None, output_dir=None, targets=None):
    """
    Create ensemble models for all targets
    
    Args:
        models_dir (str, optional): Directory with base models
        output_dir (str, optional): Directory to save ensemble models
        targets (list, optional): List of targets to create ensembles for
        
    Returns:
        dict: Dictionary of ensemble models by target
    """
    if models_dir is None:
        models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
    
    if output_dir is None:
        output_dir = models_dir
    
    # If targets not specified, detect from model files
    if targets is None:
        # Find all model files
        model_files = glob.glob(os.path.join(models_dir, "nba_*_model_*.joblib"))
        
        # Extract targets from filenames
        targets = set()
        for file in model_files:
            parts = os.path.basename(file).split('_')
            if len(parts) > 1:
                targets.add(parts[1])
        
        targets = list(targets)
    
    # Create ensembles for each target
    ensembles = {}
    
    for target in targets:
        logging.info(f"Creating ensembles for target: {target}")
        
        target_ensembles = train_ensemble_models(target, models_dir, output_dir)
        
        if target_ensembles:
            ensembles[target] = target_ensembles
    
    return ensembles

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create ensemble models for NBA predictions")
    
    parser.add_argument("--target", type=str, help="Target to create ensemble for (e.g., pts, ast)")
    parser.add_argument("--models-dir", type=str, help="Directory with base models")
    parser.add_argument("--output-dir", type=str, help="Directory to save ensemble models")
    parser.add_argument("--ensemble-types", type=str, nargs="+", 
                       choices=["voting", "stacking", "all"],
                       default=["all"], help="Types of ensembles to create")
    parser.add_argument("--model-types", type=str, nargs="+",
                       choices=["random_forest", "xgboost", "gradient_boosting", "all"],
                       default=["all"], help="Types of models to include in ensembles")
    parser.add_argument("--all-targets", action="store_true", help="Create ensembles for all targets")
    
    args = parser.parse_args()
    
    # Process ensemble types
    ensemble_types = args.ensemble_types
    if "all" in ensemble_types:
        ensemble_types = ["voting", "stacking"]
    
    # Process model types
    model_types = args.model_types
    if "all" in model_types:
        model_types = ["random_forest", "xgboost", "gradient_boosting"]
    
    if args.all_targets:
        # Create ensembles for all targets
        ensembles = create_ensemble_for_all_targets(
            models_dir=args.models_dir,
            output_dir=args.output_dir
        )
        
        logging.info(f"Created ensembles for {len(ensembles)} targets")
    elif args.target:
        # Create ensembles for specific target
        ensembles = train_ensemble_models(
            target=args.target,
            models_dir=args.models_dir,
            output_dir=args.output_dir,
            ensemble_types=ensemble_types,
            model_types=model_types
        )
        
        logging.info(f"Created {len(ensembles)} ensembles for target {args.target}")
    else:
        parser.print_help()