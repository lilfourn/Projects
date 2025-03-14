import pandas as pd
import numpy as np
import joblib
import os
import logging
from datetime import datetime
import warnings
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Try to import scikit-optimize for Bayesian optimization
try:
    from skopt import BayesSearchCV
    from skopt.space import Real, Integer, Categorical
    BAYESIAN_OPT_AVAILABLE = True
except ImportError:
    BAYESIAN_OPT_AVAILABLE = False
    warnings.warn("scikit-optimize not available. Bayesian optimization not available.")

# Try to import xgboost and lightgbm
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    warnings.warn("XGBoost not available. XGBoost model not available.")
    
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    warnings.warn("LightGBM not available. LightGBM model not available.")

try:
    from src.feature_engineering import create_feature_matrix, engineer_all_features
except ImportError:
    # Try importing without the src prefix
    from feature_engineering import create_feature_matrix, engineer_all_features

try:
    from src.memory_utils import ProgressLogger
except ImportError:
    # Try importing without the src prefix
    from memory_utils import ProgressLogger

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_training_data(data_path=None):
    """
    Load training data for model building
    
    Args:
        data_path (str, optional): Path to the engineered data file
        
    Returns:
        pd.DataFrame: DataFrame containing engineered data
    """
    try:
        if data_path is None:
            # Try to find the latest engineered data file
            current_date = datetime.now().strftime("%Y%m%d")
            engineered_dir = "/Users/lukesmac/Projects/nbaModel/data/engineered"
            
            # If the directory doesn't exist, create it
            os.makedirs(engineered_dir, exist_ok=True)
            
            # Path for today's engineered data
            engineered_path = os.path.join(engineered_dir, f"engineered_nba_data_{current_date}.csv")
            
            # If today's file doesn't exist, try to find the latest one
            if not os.path.exists(engineered_path):
                engineered_files = []
                if os.path.exists(engineered_dir):
                    engineered_files = sorted(
                        [f for f in os.listdir(engineered_dir) if f.startswith("engineered_nba_data_")],
                        reverse=True
                    )
                
                if engineered_files:
                    engineered_path = os.path.join(engineered_dir, engineered_files[0])
                else:
                    # If no engineered data is found, try to use the processed data
                    processed_dir = "/Users/lukesmac/Projects/nbaModel/data/processed"
                    
                    if os.path.exists(processed_dir):
                        processed_files = sorted(
                            [f for f in os.listdir(processed_dir) if f.startswith("processed_nba_data_")],
                            reverse=True
                        )
                        
                        if processed_files:
                            processed_path = os.path.join(processed_dir, processed_files[0])
                            logging.info(f"No engineered data found. Using processed data: {processed_path}")
                            
                            # Load and engineer the data
                            processed_data = pd.read_csv(processed_path)
                            engineered_data = engineer_all_features(processed_data)
                        
                            # Save the engineered data
                            engineered_path = os.path.join(engineered_dir, f"engineered_nba_data_{current_date}.csv")
                            engineered_data.to_csv(engineered_path, index=False)
                            
                            return engineered_data
                        else:
                            logging.error("No processed data found")
                            return None
                    else:
                        logging.error("No processed data directory found")
                        return None
            
            data_path = engineered_path
        
        # Load the data
        try:
            data = pd.read_csv(data_path)
            logging.info(f"Loaded training data with {data.shape[0]} rows and {data.shape[1]} columns")
            return data
        except Exception as e:
            logging.error(f"Error loading training data: {str(e)}")
            return None
        
    except Exception as e:
        logging.error(f"Error in load_training_data: {str(e)}")
        return None

def train_model(X_train, y_train, model_type='random_forest', hyperparams=None, low_resource_mode=False,
               target_specific_models=None, use_ensemble_stacking=False, use_linear_blend=True,
               time_gap=3):
    """
    Train a regression model
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.DataFrame): Training targets
        model_type (str): Type of model to train ('decision_tree', 'random_forest', 'gradient_boosting', 
                         'xgboost', 'lightgbm', 'stacked', 'target_specific')
        hyperparams (dict, optional): Hyperparameters for the model
        low_resource_mode (bool, optional): Whether to use efficient parameters to reduce CPU usage
        target_specific_models (dict, optional): Dictionary mapping target names to model types for target_specific model
        use_ensemble_stacking (bool): Whether to combine the selected model type with stacking
        use_linear_blend (bool): Whether to use a linear blender (Ridge) in stacked models
        time_gap (int): Gap size for TimeSeriesSplit to prevent data leakage when training
        
    Returns:
        tuple: (Trained model, feature importance)
    """
    # Default hyperparameters based on model type
    if hyperparams is None:
        if model_type == 'decision_tree':
            hyperparams = {
                'max_depth': 6 if low_resource_mode else 8,
                'min_samples_split': 10,
                'min_samples_leaf': 5,
                'random_state': 42
            }
        elif model_type == 'random_forest':
            hyperparams = {
                'n_estimators': 25 if low_resource_mode else 50,
                'max_depth': 8 if low_resource_mode else 10,
                'min_samples_split': 6,
                'min_samples_leaf': 3,
                'random_state': 42,
                'n_jobs': 1  # Only use 1 parallel job to avoid overloading
            }
        elif model_type == 'gradient_boosting':
            hyperparams = {
                'n_estimators': 25 if low_resource_mode else 50,
                'max_depth': 3 if low_resource_mode else 4,
                'learning_rate': 0.1,
                'random_state': 42,
                'subsample': 0.7 if low_resource_mode else 0.8  # More aggressive subsampling in low resource mode
            }
        elif model_type == 'xgboost' and XGBOOST_AVAILABLE:
            hyperparams = {
                'n_estimators': 25 if low_resource_mode else 50,
                'max_depth': 3 if low_resource_mode else 4,
                'learning_rate': 0.1,
                'random_state': 42,
                'subsample': 0.7 if low_resource_mode else 0.8,
                'colsample_bytree': 0.8,
                'n_jobs': 1  # Only use 1 parallel job
            }
        elif model_type == 'lightgbm' and LIGHTGBM_AVAILABLE:
            hyperparams = {
                'n_estimators': 25 if low_resource_mode else 50,
                'max_depth': 3 if low_resource_mode else 4,
                'learning_rate': 0.1,
                'random_state': 42,
                'subsample': 0.7 if low_resource_mode else 0.8,
                'colsample_bytree': 0.8,
                'n_jobs': 1  # Only use 1 parallel job
            }
        elif model_type == 'stacked':
            # Stacked model will use its own hyperparameters
            hyperparams = {}
        elif model_type == 'target_specific':
            # Target-specific models will use their own hyperparameters
            hyperparams = {}
    
    logging.info(f"Training {model_type} model with hyperparameters: {hyperparams}")
    
    # Check if it's multi-output
    is_multi_output = isinstance(y_train, (pd.DataFrame, np.ndarray)) and y_train.shape[1] > 1
    
    # Check if we should use ensemble stacking for the chosen model type
    if use_ensemble_stacking and model_type not in ['stacked', 'target_specific']:
        logging.info(f"Using ensemble stacking with {model_type} as base model")
        
        # Use specialized stacked ensemble with emphasis on the chosen model type
        model = create_specialized_ensemble(X_train, y_train, specialized_model=model_type,
                                          low_resource_mode=low_resource_mode, 
                                          use_linear_blend=use_linear_blend,
                                          hyperparams=hyperparams,
                                          time_gap=time_gap)
        
        # If multi-output, wrap in MultiOutputRegressor
        if is_multi_output:
            model = MultiOutputRegressor(model)
            
        # Train the specialized stacked model
        with ProgressLogger(total=1, desc=f"Training specialized {model_type} ensemble", unit="model") as progress:
            model.fit(X_train, y_train)
            progress.update(1)
            
        # No feature importances for stacked models
        return model, None
    
    # Create the model based on model type
    if model_type == 'decision_tree':
        base_model = DecisionTreeRegressor(**hyperparams)
    elif model_type == 'random_forest':
        base_model = RandomForestRegressor(**hyperparams)
    elif model_type == 'gradient_boosting':
        base_model = GradientBoostingRegressor(**hyperparams)
    elif model_type == 'xgboost' and XGBOOST_AVAILABLE:
        base_model = xgb.XGBRegressor(**hyperparams)
    elif model_type == 'lightgbm' and LIGHTGBM_AVAILABLE:
        base_model = lgb.LGBMRegressor(**hyperparams)
    elif model_type == 'stacked':
        # Create a standard stacked ensemble
        model = create_stacked_ensemble(X_train, y_train, low_resource_mode=low_resource_mode, 
                                        use_linear_blend=use_linear_blend, time_gap=time_gap)
        
        # If multi-output, wrap in MultiOutputRegressor
        if is_multi_output:
            model = MultiOutputRegressor(model)
            
        # Train the stacked model
        with ProgressLogger(total=1, desc="Training stacked ensemble", unit="model") as progress:
            model.fit(X_train, y_train)
            progress.update(1)
            
        # No feature importances for stacked models
        return model, None
        
    elif model_type == 'target_specific':
        # Only usable for multi-output targets
        if not is_multi_output or not isinstance(y_train, pd.DataFrame):
            raise ValueError("Target-specific models can only be used with multi-output DataFrame targets")
            
        # Create target-specific model
        model = TargetSpecificModel(target_models=target_specific_models, fallback_model='random_forest')
        
        # Train the model
        with ProgressLogger(total=len(y_train.columns), desc="Training target-specific models", unit="target") as progress:
            model.fit(X_train, y_train)
            progress.update(len(y_train.columns))
            
        # Get feature importances
        feature_importances = model.get_feature_importance()
        
        # Average feature importances across targets
        if feature_importances:
            # Filter out None values
            valid_importances = [imp for imp in feature_importances.values() if imp is not None]
            
            if valid_importances:
                # Average across targets
                feature_importance = np.mean(valid_importances, axis=0)
            else:
                feature_importance = None
        else:
            feature_importance = None
            
        return model, feature_importance
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # If we have a multi-output target, use MultiOutputRegressor for base models
    if is_multi_output:
        model = MultiOutputRegressor(base_model)
    else:
        model = base_model
    
    # Train the model with progress logging
    with ProgressLogger(total=1, desc=f"Training {model_type}", unit="model") as progress:
        model.fit(X_train, y_train)
        progress.update(1)
    
    # Extract feature importance if available
    feature_importance = None
    if not is_multi_output and hasattr(model, 'feature_importances_'):
        feature_importance = model.feature_importances_
    elif is_multi_output and hasattr(model.estimators_[0], 'feature_importances_'):
        # Average feature importance across all estimators
        importance_per_estimator = [est.feature_importances_ for est in model.estimators_]
        feature_importance = np.mean(importance_per_estimator, axis=0)
    
    logging.info("Model training complete")
    return model, feature_importance

# Custom class for target-specific models
class TargetSpecificModel(BaseEstimator, RegressorMixin):
    """
    A model that selects the best model type for each target variable
    based on their known characteristics.
    """
    def __init__(self, target_models=None, fallback_model='random_forest'):
        """
        Initialize with target-specific model types
        
        Args:
            target_models (dict): Dictionary mapping target names to model types
            fallback_model (str): Model type to use for targets not in target_models
        """
        self.target_models = target_models or {}
        self.fallback_model = fallback_model
        self.models = {}
        
    def fit(self, X, y):
        """
        Fit a model for each target
        
        Args:
            X (pd.DataFrame): Feature data
            y (pd.DataFrame): Target data
        """
        if not isinstance(y, pd.DataFrame):
            raise ValueError("Target data must be a DataFrame")
        
        for col in y.columns:
            # Get model type for this target
            model_type = self.target_models.get(col, self.fallback_model)
            
            # Create and fit model for this target
            if model_type == 'decision_tree':
                model = DecisionTreeRegressor(max_depth=8, min_samples_split=10, random_state=42)
            elif model_type == 'random_forest':
                model = RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42, n_jobs=1)
            elif model_type == 'gradient_boosting':
                model = GradientBoostingRegressor(n_estimators=50, max_depth=4, learning_rate=0.1, random_state=42)
            elif model_type == 'xgboost' and XGBOOST_AVAILABLE:
                model = xgb.XGBRegressor(n_estimators=50, max_depth=4, learning_rate=0.1, random_state=42, n_jobs=1)
            elif model_type == 'lightgbm' and LIGHTGBM_AVAILABLE:
                model = lgb.LGBMRegressor(n_estimators=50, max_depth=4, learning_rate=0.1, random_state=42, n_jobs=1)
            else:
                # Fallback to random forest
                model = RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42, n_jobs=1)
                
            # Fit model to this target
            model.fit(X, y[col])
            self.models[col] = model
            
        return self
    
    def predict(self, X):
        """
        Predict using each target-specific model
        
        Args:
            X (pd.DataFrame): Feature data
            
        Returns:
            np.ndarray: Predictions with shape (n_samples, n_targets)
        """
        if not self.models:
            raise ValueError("Models not fitted yet")
            
        # Get predictions for each target
        predictions = []
        for col, model in self.models.items():
            pred = model.predict(X)
            predictions.append(pred)
            
        # Combine into a 2D array
        return np.column_stack(predictions)
    
    def get_params(self, deep=True):
        """Return parameters for this estimator"""
        return {
            "target_models": self.target_models,
            "fallback_model": self.fallback_model
        }
    
    def set_params(self, **parameters):
        """Set parameters for this estimator"""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    
    def get_feature_importance(self, target=None):
        """
        Get feature importance for a specific target or all targets
        
        Args:
            target (str, optional): Target name to get feature importance for
            
        Returns:
            dict: Dictionary mapping targets to feature importances
        """
        if not self.models:
            raise ValueError("Models not fitted yet")
            
        if target:
            if target not in self.models:
                raise ValueError(f"Target {target} not found")
            
            model = self.models[target]
            if hasattr(model, 'feature_importances_'):
                return {target: model.feature_importances_}
            else:
                return {target: None}
        else:
            # Get feature importance for all targets
            importances = {}
            for target, model in self.models.items():
                if hasattr(model, 'feature_importances_'):
                    importances[target] = model.feature_importances_
                else:
                    importances[target] = None
            
            return importances
        
# Create a stacked ensemble model
def create_stacked_ensemble(X_train, y_train, low_resource_mode=False, use_linear_blend=True, time_gap=1):
    """
    Create a stacked ensemble of models with multiple model types
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.DataFrame or pd.Series): Training targets
        low_resource_mode (bool): Whether to use low resource mode
        use_linear_blend (bool): Whether to use a linear blender (Ridge) or more complex blender
        time_gap (int): Gap size for TimeSeriesSplit to prevent data leakage
        
    Returns:
        StackingRegressor: Stacked ensemble model
    """
    # Base estimators
    estimators = []
    
    # Add decision tree
    dt_params = {
        'max_depth': 6 if low_resource_mode else 8,
        'min_samples_split': 10,
        'min_samples_leaf': 4,
        'random_state': 42
    }
    estimators.append(('dt', DecisionTreeRegressor(**dt_params)))
    
    # Add random forest with different hyperparameters than decision tree
    rf_params = {
        'n_estimators': 25 if low_resource_mode else 50,
        'max_depth': 8 if low_resource_mode else 10,
        'min_samples_split': 6,
        'min_samples_leaf': 2,
        'max_features': 'sqrt',  # Randomize features to increase diversity
        'random_state': 42,
        'n_jobs': 1  # Use only 1 job per estimator to avoid overloading CPU
    }
    estimators.append(('rf', RandomForestRegressor(**rf_params)))
    
    # Add gradient boosting
    gb_params = {
        'n_estimators': 25 if low_resource_mode else 50,
        'max_depth': 3 if low_resource_mode else 4,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'random_state': 42
    }
    estimators.append(('gb', GradientBoostingRegressor(**gb_params)))
    
    # Add XGBoost if available - with different hyperparameters from GBM
    if XGBOOST_AVAILABLE:
        xgb_params = {
            'n_estimators': 25 if low_resource_mode else 50,
            'max_depth': 4 if low_resource_mode else 5,
            'learning_rate': 0.05,  # Different learning rate
            'colsample_bytree': 0.85,
            'subsample': 0.75,  # Different subsample ratio
            'random_state': 42,
            'n_jobs': 1  # Use only 1 job per estimator
        }
        estimators.append(('xgb', xgb.XGBRegressor(**xgb_params)))
    
    # Add LightGBM if available - with different hyperparameters
    if LIGHTGBM_AVAILABLE:
        lgb_params = {
            'n_estimators': 25 if low_resource_mode else 50,
            'max_depth': 5 if low_resource_mode else 6,  # Different depth
            'learning_rate': 0.08,  # Different learning rate
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': 1  # Use only 1 job per estimator
        }
        estimators.append(('lgb', lgb.LGBMRegressor(**lgb_params)))
    
    # Add a regularized linear model for diversity
    linear_params = {
        'alpha': 0.1,
        'random_state': 42
    }
    estimators.append(('ridge', Ridge(**linear_params)))
    
    # Final estimator (blender) - choose between simple Ridge or another model
    if use_linear_blend:
        # Simple linear blender with ridge regression
        final_estimator = Ridge(alpha=1.0)
        logging.info("Using Ridge regression as final blender in stacked ensemble")
    else:
        # More complex blender using gradient boosting
        final_estimator = GradientBoostingRegressor(
            n_estimators=25,
            max_depth=3,
            learning_rate=0.1,
            random_state=42
        )
        logging.info("Using Gradient Boosting as final blender in stacked ensemble")
    
    # Create the stacked model with time-series aware cross-validation
    cv = 3 if low_resource_mode else 5
    # Use TimeSeriesSplit for stacking to maintain chronological order
    cv_splitter = TimeSeriesSplit(n_splits=cv, gap=time_gap)
    
    stacked_model = StackingRegressor(
        estimators=estimators,
        final_estimator=final_estimator,
        cv=cv_splitter,  # Use time series split for proper validation
        n_jobs=1,  # Control parallelism
        passthrough=True  # Include original features alongside meta-features
    )
    
    return stacked_model

def tune_model_hyperparameters(X_train, y_train, model_type='random_forest', cv=5, use_bayesian=True, gap_size=3, 
                      n_iter_bayesian=20, randomized_fraction=0.5):
    """
    Tune hyperparameters for the model using grid search or Bayesian optimization
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.DataFrame): Training targets
        model_type (str): Type of model to tune ('decision_tree', 'random_forest', 'gradient_boosting', 'xgboost', 'lightgbm', 'stacked')
        cv (int, optional): Number of cross-validation folds
        use_bayesian (bool): Whether to use Bayesian optimization (if available)
        gap_size (int): Gap size for TimeSeriesSplit to prevent data leakage
        n_iter_bayesian (int): Number of iterations for Bayesian optimization
        randomized_fraction (float): Fraction of parameter space to explore randomly before Bayesian optimization
        
    Returns:
        dict: Best hyperparameters
    """
    logging.info(f"Tuning {model_type} hyperparameters with {cv}-fold cross-validation")
    
    # Use TimeSeriesSplit for time-based validation with a gap
    is_multioutput = isinstance(y_train, (pd.DataFrame, np.ndarray)) and y_train.shape[1] > 1
    
    # Create a TimeSeriesSplit with a configurable gap for time-series validation
    # A larger gap prevents data leakage by ensuring chronological separation between train and test folds
    time_series_cv = TimeSeriesSplit(n_splits=cv, gap=gap_size)
    logging.info(f"Using TimeSeriesSplit with {cv} folds and gap size of {gap_size}")
    
    # Define parameter grid or space based on model type
    if model_type == 'decision_tree':
        base_model = DecisionTreeRegressor(random_state=42)
        
        if use_bayesian and BAYESIAN_OPT_AVAILABLE:
            param_space = {
                'estimator__max_depth': Integer(3, 12),
                'estimator__min_samples_split': Integer(2, 20),
                'estimator__min_samples_leaf': Integer(1, 10),
                'estimator__criterion': Categorical(['squared_error', 'friedman_mse', 'absolute_error'])
            } if is_multioutput else {
                'max_depth': Integer(3, 12),
                'min_samples_split': Integer(2, 20),
                'min_samples_leaf': Integer(1, 10),
                'criterion': Categorical(['squared_error', 'friedman_mse', 'absolute_error'])
            }
        else:
            param_grid = {
                'estimator__max_depth': [4, 6, 8, 10],
                'estimator__min_samples_split': [2, 5, 10, 15],
                'estimator__min_samples_leaf': [1, 2, 4],
                'estimator__criterion': ['squared_error', 'friedman_mse', 'absolute_error']
            } if is_multioutput else {
                'max_depth': [4, 6, 8, 10],
                'min_samples_split': [2, 5, 10, 15],
                'min_samples_leaf': [1, 2, 4],
                'criterion': ['squared_error', 'friedman_mse', 'absolute_error']
            }
            
    elif model_type == 'random_forest':
        base_model = RandomForestRegressor(random_state=42, n_jobs=1)
        
        if use_bayesian and BAYESIAN_OPT_AVAILABLE:
            param_space = {
                'estimator__n_estimators': Integer(20, 100),
                'estimator__max_depth': Integer(4, 12),
                'estimator__min_samples_split': Integer(2, 15),
                'estimator__min_samples_leaf': Integer(1, 6)
            } if is_multioutput else {
                'n_estimators': Integer(20, 100),
                'max_depth': Integer(4, 12),
                'min_samples_split': Integer(2, 15),
                'min_samples_leaf': Integer(1, 6)
            }
        else:
            param_grid = {
                'estimator__n_estimators': [25, 50, 75],
                'estimator__max_depth': [6, 8, 10],
                'estimator__min_samples_split': [5, 10],
                'estimator__min_samples_leaf': [2, 4]
            } if is_multioutput else {
                'n_estimators': [25, 50, 75],
                'max_depth': [6, 8, 10],
                'min_samples_split': [5, 10],
                'min_samples_leaf': [2, 4]
            }
            
    elif model_type == 'gradient_boosting':
        base_model = GradientBoostingRegressor(random_state=42, subsample=0.8)
        
        if use_bayesian and BAYESIAN_OPT_AVAILABLE:
            param_space = {
                'estimator__n_estimators': Integer(20, 100),
                'estimator__max_depth': Integer(2, 6),
                'estimator__learning_rate': Real(0.01, 0.3, prior='log-uniform'),
                'estimator__subsample': Real(0.5, 1.0)
            } if is_multioutput else {
                'n_estimators': Integer(20, 100),
                'max_depth': Integer(2, 6),
                'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
                'subsample': Real(0.5, 1.0)
            }
        else:
            param_grid = {
                'estimator__n_estimators': [25, 50, 75],
                'estimator__max_depth': [3, 4, 5],
                'estimator__learning_rate': [0.05, 0.1, 0.2],
                'estimator__subsample': [0.7, 0.8, 0.9]
            } if is_multioutput else {
                'n_estimators': [25, 50, 75],
                'max_depth': [3, 4, 5],
                'learning_rate': [0.05, 0.1, 0.2],
                'subsample': [0.7, 0.8, 0.9]
            }
            
    elif model_type == 'xgboost' and XGBOOST_AVAILABLE:
        base_model = xgb.XGBRegressor(random_state=42, n_jobs=1)
        
        if use_bayesian and BAYESIAN_OPT_AVAILABLE:
            param_space = {
                'estimator__n_estimators': Integer(20, 100),
                'estimator__max_depth': Integer(2, 6),
                'estimator__learning_rate': Real(0.01, 0.3, prior='log-uniform'),
                'estimator__subsample': Real(0.5, 1.0),
                'estimator__colsample_bytree': Real(0.5, 1.0)
            } if is_multioutput else {
                'n_estimators': Integer(20, 100),
                'max_depth': Integer(2, 6),
                'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
                'subsample': Real(0.5, 1.0),
                'colsample_bytree': Real(0.5, 1.0)
            }
        else:
            param_grid = {
                'estimator__n_estimators': [25, 50, 75],
                'estimator__max_depth': [3, 4, 5],
                'estimator__learning_rate': [0.05, 0.1, 0.2],
                'estimator__subsample': [0.7, 0.8, 0.9],
                'estimator__colsample_bytree': [0.7, 0.8, 0.9]
            } if is_multioutput else {
                'n_estimators': [25, 50, 75],
                'max_depth': [3, 4, 5],
                'learning_rate': [0.05, 0.1, 0.2],
                'subsample': [0.7, 0.8, 0.9],
                'colsample_bytree': [0.7, 0.8, 0.9]
            }
            
    elif model_type == 'lightgbm' and LIGHTGBM_AVAILABLE:
        base_model = lgb.LGBMRegressor(random_state=42, n_jobs=1)
        
        if use_bayesian and BAYESIAN_OPT_AVAILABLE:
            param_space = {
                'estimator__n_estimators': Integer(20, 100),
                'estimator__max_depth': Integer(2, 6),
                'estimator__learning_rate': Real(0.01, 0.3, prior='log-uniform'),
                'estimator__subsample': Real(0.5, 1.0),
                'estimator__colsample_bytree': Real(0.5, 1.0)
            } if is_multioutput else {
                'n_estimators': Integer(20, 100),
                'max_depth': Integer(2, 6),
                'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
                'subsample': Real(0.5, 1.0),
                'colsample_bytree': Real(0.5, 1.0)
            }
        else:
            param_grid = {
                'estimator__n_estimators': [25, 50, 75],
                'estimator__max_depth': [3, 4, 5],
                'estimator__learning_rate': [0.05, 0.1, 0.2],
                'estimator__subsample': [0.7, 0.8, 0.9],
                'estimator__colsample_bytree': [0.7, 0.8, 0.9]
            } if is_multioutput else {
                'n_estimators': [25, 50, 75],
                'max_depth': [3, 4, 5],
                'learning_rate': [0.05, 0.1, 0.2],
                'subsample': [0.7, 0.8, 0.9],
                'colsample_bytree': [0.7, 0.8, 0.9]
            }
            
    elif model_type == 'stacked':
        # Create a stacked ensemble with default parameters
        base_model = create_stacked_ensemble(X_train, y_train, low_resource_mode=True)
        
        # Limited parameter tuning for stacked model
        if use_bayesian and BAYESIAN_OPT_AVAILABLE:
            param_space = {
                'estimator__final_estimator__alpha': Real(0.01, 10.0, prior='log-uniform')
            } if is_multioutput else {
                'final_estimator__alpha': Real(0.01, 10.0, prior='log-uniform')
            }
        else:
            param_grid = {
                'estimator__final_estimator__alpha': [0.1, 1.0, 5.0, 10.0]
            } if is_multioutput else {
                'final_estimator__alpha': [0.1, 1.0, 5.0, 10.0]
            }
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Create model for optimization
    if is_multioutput:
        model = MultiOutputRegressor(base_model)
    else:
        model = base_model
    
    # Use Bayesian optimization if available and requested
    if use_bayesian and BAYESIAN_OPT_AVAILABLE:
        search = BayesSearchCV(
            model,
            param_space,
            n_iter=n_iter_bayesian,  # Configurable number of iterations
            cv=time_series_cv,
            scoring='neg_mean_squared_error',
            n_jobs=1,  # Use only 1 job to avoid overloading
            verbose=2,
            random_state=42,
            n_initial_points=int(n_iter_bayesian * randomized_fraction)  # Start with random sampling
        )
        logging.info(f"Using Bayesian optimization for hyperparameter tuning with {n_iter_bayesian} iterations")
        logging.info(f"Starting with {int(n_iter_bayesian * randomized_fraction)} random samples before Bayesian optimization")
    else:
        from sklearn.model_selection import RandomizedSearchCV
        # Use RandomizedSearchCV instead of GridSearchCV for more efficient search
        if len(param_grid) > 5:  # If we have many parameters, use randomized search
            # Convert param_grid to param_distributions for RandomizedSearchCV
            search = RandomizedSearchCV(
                model,
                param_grid,
                n_iter=min(20, np.prod([len(values) for values in param_grid.values()])),  # Reasonable number of iterations
                cv=time_series_cv,
                scoring='neg_mean_squared_error',
                verbose=1,
                n_jobs=1,  # Use only 1 job to avoid overloading
                random_state=42
            )
            logging.info("Using randomized search for hyperparameter tuning")
        else:
            search = GridSearchCV(
                model,
                param_grid,
                cv=time_series_cv,
                scoring='neg_mean_squared_error',
                verbose=1,
                n_jobs=1  # Use only 1 job to avoid overloading
            )
            logging.info("Using grid search for hyperparameter tuning")
    
    # Show progress bar if available
    if use_bayesian and BAYESIAN_OPT_AVAILABLE:
        total_fits = cv * n_iter_bayesian  # Bayesian optimization iterations
    elif 'RandomizedSearchCV' in str(type(search)):
        total_fits = cv * search.n_iter  # Randomized search iterations
    else:
        # For grid search, calculate total number of parameter combinations
        total_fits = cv * np.prod([len(values) for values in param_grid.values()])
    
    with ProgressLogger(total=total_fits, desc=f"Tuning {model_type}", unit="fit") as progress:
        # Define a callback for Bayesian search if available
        if use_bayesian and BAYESIAN_OPT_AVAILABLE:
            def on_step(optim_result):
                progress.update(1)
                return True
            
            search.callback = on_step
        
        # Fit the search
        search.fit(X_train, y_train)
    
    # Extract best parameters
    best_params = search.best_params_
    
    # Convert params from estimator__ format to direct format
    if is_multioutput:
        best_params_direct = {k.replace('estimator__', ''): v for k, v in best_params.items()}
    else:
        best_params_direct = best_params
    
    logging.info(f"Best hyperparameters for {model_type}: {best_params_direct}")
    return best_params_direct

def evaluate_model(model, X_test, y_test, feature_names=None, target_names=None, 
                 time_series_validation=True, time_gap=3, cv=5):
    """
    Evaluate the trained model on test data with optional time-series validation
    
    Args:
        model (object): Trained model
        X_test (pd.DataFrame): Test features
        y_test (pd.DataFrame): Test targets
        feature_names (list, optional): Names of features
        target_names (list, optional): Names of targets
        time_series_validation (bool): Whether to use time-series cross-validation instead of a single test set
        time_gap (int): Gap size for TimeSeriesSplit to prevent data leakage
        cv (int): Number of cross-validation folds if using time-series validation
        
    Returns:
        dict: Dictionary of evaluation metrics
    """
    logging.info("Evaluating model on test data")
    
    # Use column names if they're available and names aren't provided
    if feature_names is None and isinstance(X_test, pd.DataFrame):
        feature_names = X_test.columns.tolist()
    
    if target_names is None and isinstance(y_test, pd.DataFrame):
        target_names = y_test.columns.tolist()
    
    # If using time-series validation, evaluate across multiple chronological splits
    if time_series_validation and isinstance(X_test, pd.DataFrame) and len(X_test) > cv * 10:
        logging.info(f"Using time-series cross-validation with {cv} folds and gap={time_gap}")
        
        # Create a TimeSeriesSplit with a gap between train and test sets
        tscv = TimeSeriesSplit(n_splits=cv, gap=time_gap)
        
        # Sort data by index if it's DateTime indexed
        if hasattr(X_test, 'index') and isinstance(X_test.index, pd.DatetimeIndex):
            logging.info("Sorting data chronologically for time-series validation")
            # Combine X and y for consistent splits, ensuring chronological order
            combined_data = pd.concat([X_test.reset_index(drop=True), 
                                      y_test.reset_index(drop=True)], axis=1)
            combined_data = combined_data.sort_index()
            
            # Split back into X and y
            feature_cols = X_test.columns
            X_test = combined_data[feature_cols]
            y_test = combined_data.drop(columns=feature_cols)
        
        # Store predictions and actual values for each fold
        all_y_true = []
        all_y_pred = []
        
        with ProgressLogger(total=cv, desc="Time-series validation", unit="fold") as progress:
            for train_idx, test_idx in tscv.split(X_test):
                # Get the test fold
                X_fold = X_test.iloc[test_idx]
                y_fold = y_test.iloc[test_idx]
                
                # Make predictions on this fold
                fold_pred = model.predict(X_fold)
                
                # Store actual and predicted values
                all_y_true.append(y_fold)
                all_y_pred.append(fold_pred)
                
                progress.update(1)
        
        # Combine predictions from all folds
        y_test = pd.concat(all_y_true)
        y_pred = np.vstack(all_y_pred) if len(all_y_pred[0].shape) > 1 else np.concatenate(all_y_pred)
        
        # Convert predictions to DataFrame if y_test is a DataFrame
        if isinstance(y_test, pd.DataFrame):
            y_pred = pd.DataFrame(y_pred, columns=y_test.columns, index=y_test.index)
            
    else:
        # Standard evaluation on a single test set
        logging.info("Evaluating on single test set")
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # If y_test is a DataFrame, convert y_pred to DataFrame with same columns
        if isinstance(y_test, pd.DataFrame):
            y_pred = pd.DataFrame(y_pred, columns=y_test.columns, index=y_test.index)
    
    # Calculate metrics for each target variable
    metrics = {}
    
    # If we have multiple target variables (multi-output)
    if isinstance(y_test, (pd.DataFrame, np.ndarray)) and y_test.shape[1] > 1:
        overall_mse = mean_squared_error(y_test, y_pred)
        overall_mae = mean_absolute_error(y_test, y_pred)
        overall_r2 = r2_score(y_test, y_pred)
        
        metrics['overall'] = {
            'mse': overall_mse,
            'rmse': np.sqrt(overall_mse),
            'mae': overall_mae,
            'r2': overall_r2
        }
        
        # Per-target metrics
        for i, target in enumerate(target_names):
            target_y_test = y_test.iloc[:, i] if isinstance(y_test, pd.DataFrame) else y_test[:, i]
            target_y_pred = y_pred.iloc[:, i] if isinstance(y_pred, pd.DataFrame) else y_pred[:, i]
            
            mse = mean_squared_error(target_y_test, target_y_pred)
            mae = mean_absolute_error(target_y_test, target_y_pred)
            r2 = r2_score(target_y_test, target_y_pred)
            
            metrics[target] = {
                'mse': mse,
                'rmse': np.sqrt(mse),
                'mae': mae,
                'r2': r2
            }
    else:
        # Single target variable
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        target_name = target_names[0] if target_names else 'target'
        
        metrics[target_name] = {
            'mse': mse,
            'rmse': np.sqrt(mse),
            'mae': mae,
            'r2': r2
        }
    
    # Log the evaluation results
    for target, target_metrics in metrics.items():
        logging.info(f"Metrics for {target}:")
        for metric_name, metric_value in target_metrics.items():
            logging.info(f"  {metric_name}: {metric_value:.4f}")
    
    return metrics

def save_model(model, output_dir=None, model_name=None):
    """
    Save the trained model to disk
    
    Args:
        model (object): Trained model
        output_dir (str, optional): Directory to save the model
        model_name (str, optional): Name of the model file
        
    Returns:
        str: Path to the saved model
    """
    if output_dir is None:
        output_dir = "/Users/lukesmac/Projects/nbaModel/models"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    if model_name is None:
        current_date = datetime.now().strftime("%Y%m%d")
        model_name = f"nba_dt_model_{current_date}.joblib"
    
    model_path = os.path.join(output_dir, model_name)
    
    try:
        joblib.dump(model, model_path)
        logging.info(f"Model saved to {model_path}")
        return model_path
    except Exception as e:
        logging.error(f"Error saving model: {str(e)}")
        return None

def analyze_feature_importance(model, X, y, feature_names, output_dir=None, low_resource_mode=False,
                           permutation_importance_only=False):
    """
    Analyze and save feature importance using both model-based and permutation importance
    
    Args:
        model (object): Trained model
        X (pd.DataFrame): Feature matrix
        y (pd.DataFrame): Target matrix
        feature_names (list): List of feature names
        output_dir (str, optional): Directory to save the results
        low_resource_mode (bool, optional): Whether to use smaller data samples for permutation importance
        permutation_importance_only (bool): Whether to rely only on permutation importance for better reliability
        
    Returns:
        dict: Feature importance analysis results
    """
    logging.info("Analyzing feature importance...")
    
    # Initialize results dictionary
    importance_results = {}
    
    # Extract model-based feature importance if not using permutation_importance_only
    if not permutation_importance_only:
        logging.info("Extracting model-based feature importance")
        
        # If we already have feature importances from the model
        if hasattr(model, 'feature_importances_'):
            # For single output models with feature_importances_ attribute
            importances = model.feature_importances_
            importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
            importance_df = importance_df.sort_values('importance', ascending=False)
            importance_results['model_importance'] = importance_df.to_dict('records')
            
        elif hasattr(model, 'estimators_') and hasattr(model.estimators_[0], 'feature_importances_'):
            # For multi-output models
            all_importances = []
            for i, estimator in enumerate(model.estimators_):
                if hasattr(estimator, 'feature_importances_'):
                    importances = estimator.feature_importances_
                    target_name = y.columns[i] if isinstance(y, pd.DataFrame) else f"target_{i}"
                    importance_df = pd.DataFrame({
                        'feature': feature_names, 
                        'importance': importances,
                        'target': target_name
                    })
                    all_importances.append(importance_df)
            
            if all_importances:
                # Combine all target importances
                combined_importance = pd.concat(all_importances)
                # Get top features per target
                top_features_per_target = {}
                for target in combined_importance['target'].unique():
                    target_imp = combined_importance[combined_importance['target'] == target]
                    target_imp = target_imp.sort_values('importance', ascending=False)
                    top_features_per_target[target] = target_imp.head(15).to_dict('records')
                
                importance_results['target_importances'] = top_features_per_target
    else:
        logging.info("Skipping model-based feature importance in favor of permutation importance only")
    
    # Perform permutation importance for more reliable measure
    logging.info("Calculating permutation importance for more reliable feature importance")
    try:
        # If low resource mode, sample a subset of data for permutation importance
        if low_resource_mode and len(X) > 1000:
            # Take a sample of 1000 rows or 20% of data, whichever is smaller
            sample_size = min(1000, int(len(X) * 0.2))
            sample_indices = np.random.choice(len(X), size=sample_size, replace=False)
            X_sample = X.iloc[sample_indices] if isinstance(X, pd.DataFrame) else X[sample_indices]
            
            if isinstance(y, pd.DataFrame):
                y_sample = y.iloc[sample_indices]
            else:
                y_sample = y[sample_indices]
                
            logging.info(f"Using reduced sample of {sample_size} rows for permutation importance (low resource mode)")
        else:
            X_sample = X
            y_sample = y
            
        # If X is a DataFrame, make sure to preserve feature names
        if isinstance(X_sample, pd.DataFrame):
            feature_names_sample = X_sample.columns.tolist()
        else:
            feature_names_sample = feature_names
            
        # For multi-output targets, analyze each target separately
        if isinstance(y_sample, pd.DataFrame) and y_sample.shape[1] > 1:
            perm_importance_results = {}
            
            for i, col in enumerate(y_sample.columns):
                # Get the target column
                target = y_sample[col]
                
                # Get the corresponding estimator for multi-output models
                if hasattr(model, 'estimators_'):
                    estimator = model.estimators_[i]
                    
                    # Calculate permutation importance with reduced repeats and cores for efficiency
                    n_repeats = 3 if low_resource_mode else 5
                    with ProgressLogger(total=1, desc=f"Permutation importance for {col}", unit="target") as progress:
                        perm_importance = permutation_importance(
                            estimator, X_sample, target, n_repeats=n_repeats, random_state=42, n_jobs=2
                        )
                        progress.update(1)
                    
                    # Create dataframe for this target
                    perm_imp_df = pd.DataFrame({
                        'feature': feature_names_sample,
                        'importance_mean': perm_importance.importances_mean,
                        'importance_std': perm_importance.importances_std
                    })
                    perm_imp_df = perm_imp_df.sort_values('importance_mean', ascending=False)
                    
                    # Store more comprehensive results
                    perm_importance_results[col] = {
                        'top_features': perm_imp_df.head(15).to_dict('records'),
                        'all_features': perm_imp_df.to_dict('records'),
                        'summary': {
                            'top_feature': perm_imp_df.iloc[0]['feature'] if not perm_imp_df.empty else None,
                            'top_importance': perm_imp_df.iloc[0]['importance_mean'] if not perm_imp_df.empty else None,
                            'mean_importance': perm_imp_df['importance_mean'].mean() if not perm_imp_df.empty else None
                        }
                    }
            
            importance_results['permutation_importance'] = perm_importance_results
        else:
            # Single target case with reduced repeats and cores
            n_repeats = 3 if low_resource_mode else 5
            with ProgressLogger(total=1, desc="Calculating permutation importance", unit="model") as progress:
                perm_importance = permutation_importance(
                    model, X_sample, y_sample, n_repeats=n_repeats, random_state=42, n_jobs=2
                )
                progress.update(1)
            
            perm_imp_df = pd.DataFrame({
                'feature': feature_names_sample,
                'importance_mean': perm_importance.importances_mean,
                'importance_std': perm_importance.importances_std
            })
            perm_imp_df = perm_imp_df.sort_values('importance_mean', ascending=False)
            
            # More comprehensive results for single target case
            importance_results['permutation_importance'] = {
                'top_features': perm_imp_df.head(20).to_dict('records'),
                'all_features': perm_imp_df.to_dict('records'),
                'summary': {
                    'top_feature': perm_imp_df.iloc[0]['feature'] if not perm_imp_df.empty else None,
                    'top_importance': perm_imp_df.iloc[0]['importance_mean'] if not perm_imp_df.empty else None,
                    'mean_importance': perm_imp_df['importance_mean'].mean() if not perm_imp_df.empty else None
                }
            }
            
    except Exception as e:
        logging.warning(f"Could not calculate permutation importance: {e}")
    
    # Save results if output directory is provided
    if output_dir is not None:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save to file
        current_date = datetime.now().strftime("%Y%m%d")
        importance_path = os.path.join(output_dir, f"feature_importance_{current_date}.json")
        
        try:
            import json
            with open(importance_path, 'w') as f:
                json.dump(importance_results, f, indent=4)
            logging.info(f"Feature importance saved to {importance_path}")
        except Exception as e:
            logging.error(f"Error saving feature importance: {str(e)}")
    
    return importance_results

# Function to create a specialized stacked ensemble with a specific model type emphasized
def create_specialized_ensemble(X_train, y_train, specialized_model='random_forest', 
                              low_resource_mode=False, use_linear_blend=True,
                              hyperparams=None, time_gap=1):
    """
    Create a stacked ensemble with emphasis on a specific model type
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.DataFrame or pd.Series): Training targets
        specialized_model (str): The model type to emphasize in the ensemble
        low_resource_mode (bool): Whether to use low resource mode
        use_linear_blend (bool): Whether to use a linear blender
        hyperparams (dict): Hyperparameters for the specialized model
        time_gap (int): Gap size for TimeSeriesSplit
        
    Returns:
        StackingRegressor: Specialized stacked ensemble model
    """
    logging.info(f"Creating specialized ensemble with focus on {specialized_model}")
    
    # Base estimators
    estimators = []
    
    # Default hyperparameters if none provided
    if hyperparams is None:
        if specialized_model == 'decision_tree':
            hyperparams = {
                'max_depth': 8,
                'min_samples_split': 10,
                'min_samples_leaf': 4,
                'random_state': 42
            }
        elif specialized_model == 'random_forest':
            hyperparams = {
                'n_estimators': 50,
                'max_depth': 10,
                'min_samples_split': 6,
                'min_samples_leaf': 3,
                'random_state': 42,
                'n_jobs': 1
            }
        elif specialized_model == 'gradient_boosting':
            hyperparams = {
                'n_estimators': 50,
                'max_depth': 4,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'random_state': 42
            }
        elif specialized_model == 'xgboost' and XGBOOST_AVAILABLE:
            hyperparams = {
                'n_estimators': 50,
                'max_depth': 5,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'n_jobs': 1
            }
        elif specialized_model == 'lightgbm' and LIGHTGBM_AVAILABLE:
            hyperparams = {
                'n_estimators': 50,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'n_jobs': 1
            }
    
    # Add the specialized model (potentially multiple variants with different hyperparameters)
    if specialized_model == 'decision_tree':
        # Add the main decision tree with provided hyperparameters
        estimators.append(('dt_main', DecisionTreeRegressor(**hyperparams)))
        
        # Add variant with different hyperparameters
        dt_variant_params = hyperparams.copy()
        dt_variant_params['max_depth'] = max(hyperparams.get('max_depth', 8) - 2, 3)
        dt_variant_params['min_samples_leaf'] = hyperparams.get('min_samples_leaf', 4) * 2
        estimators.append(('dt_variant', DecisionTreeRegressor(**dt_variant_params)))
        
    elif specialized_model == 'random_forest':
        # Add the main random forest with provided hyperparameters
        estimators.append(('rf_main', RandomForestRegressor(**hyperparams)))
        
        # Add variant with different feature sampling
        rf_variant_params = hyperparams.copy()
        rf_variant_params['max_features'] = 'sqrt'
        estimators.append(('rf_variant1', RandomForestRegressor(**rf_variant_params)))
        
        # Add variant with different tree depth
        rf_variant2_params = hyperparams.copy()
        rf_variant2_params['max_depth'] = max(hyperparams.get('max_depth', 10) - 3, 3)
        estimators.append(('rf_variant2', RandomForestRegressor(**rf_variant2_params)))
        
    elif specialized_model == 'gradient_boosting':
        # Add the main gradient boosting with provided hyperparameters
        estimators.append(('gb_main', GradientBoostingRegressor(**hyperparams)))
        
        # Add variant with different learning rate
        gb_variant_params = hyperparams.copy()
        gb_variant_params['learning_rate'] = hyperparams.get('learning_rate', 0.1) / 2
        gb_variant_params['n_estimators'] = hyperparams.get('n_estimators', 50) * 2
        estimators.append(('gb_variant', GradientBoostingRegressor(**gb_variant_params)))
        
    elif specialized_model == 'xgboost' and XGBOOST_AVAILABLE:
        # Add the main XGBoost with provided hyperparameters
        estimators.append(('xgb_main', xgb.XGBRegressor(**hyperparams)))
        
        # Add variant with different hyperparameters
        xgb_variant_params = hyperparams.copy()
        xgb_variant_params['learning_rate'] = hyperparams.get('learning_rate', 0.1) / 2
        xgb_variant_params['n_estimators'] = hyperparams.get('n_estimators', 50) * 2
        estimators.append(('xgb_variant', xgb.XGBRegressor(**xgb_variant_params)))
        
    elif specialized_model == 'lightgbm' and LIGHTGBM_AVAILABLE:
        # Add the main LightGBM with provided hyperparameters
        estimators.append(('lgb_main', lgb.LGBMRegressor(**hyperparams)))
        
        # Add variant with different hyperparameters
        lgb_variant_params = hyperparams.copy()
        lgb_variant_params['learning_rate'] = hyperparams.get('learning_rate', 0.1) / 2
        lgb_variant_params['n_estimators'] = hyperparams.get('n_estimators', 50) * 2
        estimators.append(('lgb_variant', lgb.LGBMRegressor(**lgb_variant_params)))
    
    # Add other model types (with reduced complexity/size)
    if specialized_model != 'decision_tree':
        dt_params = {
            'max_depth': 6 if low_resource_mode else 8,
            'min_samples_split': 10,
            'random_state': 42
        }
        estimators.append(('dt', DecisionTreeRegressor(**dt_params)))
    
    if specialized_model != 'random_forest':
        rf_params = {
            'n_estimators': 25 if low_resource_mode else 50,
            'max_depth': 6 if low_resource_mode else 8,
            'random_state': 42,
            'n_jobs': 1
        }
        estimators.append(('rf', RandomForestRegressor(**rf_params)))
        
    if specialized_model != 'gradient_boosting':
        gb_params = {
            'n_estimators': 25 if low_resource_mode else 50,
            'max_depth': 3 if low_resource_mode else 4,
            'learning_rate': 0.1,
            'random_state': 42
        }
        estimators.append(('gb', GradientBoostingRegressor(**gb_params)))
    
    if XGBOOST_AVAILABLE and specialized_model != 'xgboost':
        xgb_params = {
            'n_estimators': 25 if low_resource_mode else 50,
            'max_depth': 3 if low_resource_mode else 4,
            'learning_rate': 0.1,
            'random_state': 42,
            'n_jobs': 1
        }
        estimators.append(('xgb', xgb.XGBRegressor(**xgb_params)))
        
    if LIGHTGBM_AVAILABLE and specialized_model != 'lightgbm':
        lgb_params = {
            'n_estimators': 25 if low_resource_mode else 50,
            'max_depth': 5 if low_resource_mode else 6,
            'learning_rate': 0.1,
            'random_state': 42,
            'n_jobs': 1
        }
        estimators.append(('lgb', lgb.LGBMRegressor(**lgb_params)))
    
    # Add a regularized linear model
    estimators.append(('ridge', Ridge(alpha=1.0, random_state=42)))
    
    # Choose the final estimator (blender)
    if use_linear_blend:
        final_estimator = Ridge(alpha=1.0)
    else:
        if specialized_model == 'gradient_boosting':
            final_estimator = GradientBoostingRegressor(
                n_estimators=25,
                max_depth=3,
                learning_rate=0.1,
                random_state=42
            )
        elif specialized_model == 'random_forest':
            final_estimator = RandomForestRegressor(
                n_estimators=25,
                max_depth=4,
                random_state=42,
                n_jobs=1
            )
        else:
            final_estimator = GradientBoostingRegressor(
                n_estimators=25,
                max_depth=3,
                learning_rate=0.1,
                random_state=42
            )
    
    # Create the stacked model with time-series aware cross-validation
    cv = 3 if low_resource_mode else 5
    cv_splitter = TimeSeriesSplit(n_splits=cv, gap=time_gap)
    
    stacked_model = StackingRegressor(
        estimators=estimators,
        final_estimator=final_estimator,
        cv=cv_splitter,
        n_jobs=1,
        passthrough=True  # Include original features
    )
    
    return stacked_model

def save_metrics(metrics, feature_names, target_names, output_dir=None, metrics_name=None):
    """
    Save model evaluation metrics to disk
    
    Args:
        metrics (dict): Dictionary of evaluation metrics
        feature_names (list): Names of features used
        target_names (list): Names of target variables
        output_dir (str, optional): Directory to save the metrics
        metrics_name (str, optional): Name of the metrics file
        
    Returns:
        str: Path to the saved metrics
    """
    if output_dir is None:
        output_dir = "/Users/lukesmac/Projects/nbaModel/models"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    if metrics_name is None:
        current_date = datetime.now().strftime("%Y%m%d")
        metrics_name = f"nba_dt_metrics_{current_date}.json"
    
    metrics_path = os.path.join(output_dir, metrics_name)
    
    # Combine metrics with feature and target information
    metrics_data = {
        'metrics': metrics,
        'feature_names': feature_names,
        'target_names': target_names,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    try:
        import json
        with open(metrics_path, 'w') as f:
            json.dump(metrics_data, f, indent=4)
        
        logging.info(f"Metrics saved to {metrics_path}")
        return metrics_path
    except Exception as e:
        logging.error(f"Error saving metrics: {str(e)}")
        return None

def train_and_evaluate_model(data_path=None, model_type='random_forest', tune_hyperparams=False, 
                          use_cv=True, train_separate_models=False, test_size=0.2, random_state=42,
                          low_resource_mode=False, use_bayesian_tuning=True, time_gap=1,
                          model_per_stat=None):
    """
    End-to-end training and evaluation of the model
    
    Args:
        data_path (str, optional): Path to the engineered data file
        model_type (str): Type of model to train ('decision_tree', 'random_forest', 'gradient_boosting', 
                         'xgboost', 'lightgbm', 'stacked', 'target_specific')
        tune_hyperparams (bool, optional): Whether to tune hyperparameters
        use_cv (bool, optional): Whether to use cross-validation
        train_separate_models (bool, optional): Whether to train separate models for each stat
        test_size (float, optional): Proportion of data to use for testing
        random_state (int, optional): Random seed for reproducibility
        low_resource_mode (bool, optional): Whether to use efficient model parameters to reduce CPU usage
        use_bayesian_tuning (bool): Whether to use Bayesian optimization for hyperparameter tuning
        time_gap (int): Gap size for TimeSeriesSplit to prevent data leakage (in samples)
        model_per_stat (dict, optional): Dictionary mapping stat names to best model types
        
    Returns:
        tuple: Trained model(s), evaluation metrics
    """
    resource_mode = "low resource mode (reduced CPU usage)" if low_resource_mode else "standard resource mode"
    logging.info(f"Starting {model_type} model training and evaluation" + 
                 f" with {'cross-validation' if use_cv else 'train/test split'}" +
                 f" and {'separate models per stat' if train_separate_models else 'multi-output model'}" +
                 f" in {resource_mode}")
    
    # Check if we're using the target-specific model type
    using_target_specific = (model_type == 'target_specific')
    
    # Default model per stat mappings if not provided but using target specific model
    if using_target_specific and model_per_stat is None:
        # Define which model works best for each stat based on R² performance
        model_per_stat = {
            'pts': 'gradient_boosting',  # Points well predicted by gradient boosting
            'reb': 'random_forest',      # Rebounds well predicted by random forest
            'ast': 'gradient_boosting',  # Assists well predicted by gradient boosting
            'stl': 'random_forest',      # Steals better with random forest
            'blk': 'xgboost' if XGBOOST_AVAILABLE else 'gradient_boosting',  # Blocks better with xgboost
            'TOV_x': 'random_forest',    # Turnovers better with random forest
            'plusMinus': 'lightgbm' if LIGHTGBM_AVAILABLE else 'gradient_boosting',  # Plus-minus better with lightgbm
            
            # Additional stat mappings
            'fgm': 'gradient_boosting',
            'fga': 'gradient_boosting',
            'tptfgm': 'gradient_boosting',
            'tptfga': 'random_forest',
            'ftm': 'gradient_boosting',
            'fta': 'gradient_boosting',
            
            # Default for any other stats
            'default': 'random_forest'
        }
    
    # Load training data
    data = load_training_data(data_path)
    
    if data is None:
        logging.error("Failed to load training data")
        return None, None
    
    # Create feature matrix and target array
    X, y, feature_names = create_feature_matrix(data)
    
    # Get target names
    if isinstance(y, pd.DataFrame):
        target_names = y.columns.tolist()
    else:
        target_names = ['target']
    
    # Log the targets being modeled
    logging.info(f"Modeling {len(target_names)} targets: {', '.join(target_names)}")
    
    # If using target-specific model but not train_separate_models, override to use the TargetSpecificModel
    if using_target_specific and not train_separate_models:
        train_separate_models = False  # Still false, but we'll use a different path
        logging.info(f"Using target-specific model with specialized models per stat")
        
        # Prepare CV split with time gap for data leakage prevention
        if use_cv:
            tscv = TimeSeriesSplit(n_splits=5, gap=time_gap)
            
            # Initialize lists to store metrics
            cv_scores = {
                'overall': {'mse': [], 'rmse': [], 'mae': [], 'r2': []}
            }
            
            # Initialize target-specific metrics
            for target in target_names:
                cv_scores[target] = {'mse': [], 'rmse': [], 'mae': [], 'r2': []}
            
            # Perform cross-validation with progress tracking
            with ProgressLogger(total=len(list(tscv.split(X))), desc="Cross-validation", unit="fold") as progress:
                for train_idx, test_idx in tscv.split(X):
                    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                    
                    # Train model with target-specific models without hyperparameter tuning in CV
                    model, _ = train_model(X_train, y_train, "target_specific", None, low_resource_mode, model_per_stat)
                    
                    # Evaluate on the test fold
                    y_pred = model.predict(X_test)
                    
                    # Calculate overall metrics
                    overall_mse = mean_squared_error(y_test, y_pred)
                    overall_mae = mean_absolute_error(y_test, y_pred)
                    overall_r2 = r2_score(y_test, y_pred)
                    
                    # Store overall metrics
                    cv_scores['overall']['mse'].append(overall_mse)
                    cv_scores['overall']['rmse'].append(np.sqrt(overall_mse))
                    cv_scores['overall']['mae'].append(overall_mae)
                    cv_scores['overall']['r2'].append(overall_r2)
                    
                    # Calculate and store target-specific metrics
                    for i, target in enumerate(target_names):
                        y_test_target = y_test.iloc[:, i]
                        y_pred_target = y_pred[:, i]
                        
                        mse = mean_squared_error(y_test_target, y_pred_target)
                        mae = mean_absolute_error(y_test_target, y_pred_target)
                        r2 = r2_score(y_test_target, y_pred_target)
                        
                        cv_scores[target]['mse'].append(mse)
                        cv_scores[target]['rmse'].append(np.sqrt(mse))
                        cv_scores[target]['mae'].append(mae)
                        cv_scores[target]['r2'].append(r2)
                    
                    progress.update(1)
            
            # Calculate average scores
            avg_metrics = {}
            for target, scores in cv_scores.items():
                avg_metrics[target] = {
                    'mse': np.mean(scores['mse']),
                    'rmse': np.mean(scores['rmse']),
                    'mae': np.mean(scores['mae']),
                    'r2': np.mean(scores['r2'])
                }
                
                # Print R² scores for each target
                logging.info(f"R² for {target}: {avg_metrics[target]['r2']:.4f}")
            
            # Train final model on all data
            # No hyperparameter tuning for target-specific model
            model, importance = train_model(X, y, "target_specific", None, low_resource_mode, model_per_stat)
            
            # Save the model and metrics
            model_path = save_model(model, model_name=f"nba_target_specific_{datetime.now().strftime('%Y%m%d')}.joblib")
            metrics_path = save_metrics(avg_metrics, feature_names, target_names)
            
            # No feature importance analysis for target-specific models
            # (already handled in train_model function)
            
            logging.info("Target-specific model training and evaluation complete")
            return model, avg_metrics
            
        else:
            # Regular train/test split with chronological ordering
            # Sort by date if available
            if 'game_date' in data.columns:
                data = data.sort_values('game_date')
                # Re-create feature matrix after sorting
                X, y, feature_names = create_feature_matrix(data)
            
            # Take the last test_size% as test data (chronological split)
            split_idx = int(len(X) * (1 - test_size))
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            
            logging.info(f"Training set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")
            
            # Train model with target-specific models
            model, _ = train_model(X_train, y_train, "target_specific", None, low_resource_mode, model_per_stat)
            
            # Evaluate the model
            metrics = evaluate_model(model, X_test, y_test, feature_names, target_names)
            
            # Save the model
            model_path = save_model(model, model_name=f"nba_target_specific_{datetime.now().strftime('%Y%m%d')}.joblib")
            
            # Save the metrics
            metrics_path = save_metrics(metrics, feature_names, target_names)
            
            logging.info("Target-specific model training and evaluation complete")
            return model, metrics
    
    elif model_type == 'stacked':
        # Train a stacked ensemble model
        logging.info("Training stacked ensemble model")
        
        # Prepare CV split with time gap for data leakage prevention
        if use_cv:
            tscv = TimeSeriesSplit(n_splits=5, gap=time_gap)
            
            # Initialize lists to store metrics
            cv_scores = {
                'overall': {'mse': [], 'rmse': [], 'mae': [], 'r2': []}
            }
            
            # Initialize target-specific metrics
            for target in target_names:
                cv_scores[target] = {'mse': [], 'rmse': [], 'mae': [], 'r2': []}
            
            # Perform cross-validation
            with ProgressLogger(total=len(list(tscv.split(X))), desc="Cross-validation", unit="fold") as progress:
                for train_idx, test_idx in tscv.split(X):
                    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                    
                    # Create and train a stacked ensemble
                    model, _ = train_model(X_train, y_train, "stacked", None, low_resource_mode)
                    
                    # Evaluate on the test fold
                    y_pred = model.predict(X_test)
                    
                    # Calculate overall metrics
                    overall_mse = mean_squared_error(y_test, y_pred)
                    overall_mae = mean_absolute_error(y_test, y_pred)
                    overall_r2 = r2_score(y_test, y_pred)
                    
                    # Store overall metrics
                    cv_scores['overall']['mse'].append(overall_mse)
                    cv_scores['overall']['rmse'].append(np.sqrt(overall_mse))
                    cv_scores['overall']['mae'].append(overall_mae)
                    cv_scores['overall']['r2'].append(overall_r2)
                    
                    # Calculate and store target-specific metrics
                    for i, target in enumerate(target_names):
                        y_test_target = y_test.iloc[:, i]
                        y_pred_target = y_pred[:, i]
                        
                        mse = mean_squared_error(y_test_target, y_pred_target)
                        mae = mean_absolute_error(y_test_target, y_pred_target)
                        r2 = r2_score(y_test_target, y_pred_target)
                        
                        cv_scores[target]['mse'].append(mse)
                        cv_scores[target]['rmse'].append(np.sqrt(mse))
                        cv_scores[target]['mae'].append(mae)
                        cv_scores[target]['r2'].append(r2)
                    
                    progress.update(1)
            
            # Calculate average scores
            avg_metrics = {}
            for target, scores in cv_scores.items():
                avg_metrics[target] = {
                    'mse': np.mean(scores['mse']),
                    'rmse': np.mean(scores['rmse']),
                    'mae': np.mean(scores['mae']),
                    'r2': np.mean(scores['r2'])
                }
                
                # Print R² scores for each target
                logging.info(f"R² for {target}: {avg_metrics[target]['r2']:.4f}")
            
            # Train final model on all data
            model, _ = train_model(X, y, "stacked", None, low_resource_mode)
            
            # Save the model and metrics
            model_path = save_model(model, model_name=f"nba_stacked_{datetime.now().strftime('%Y%m%d')}.joblib")
            metrics_path = save_metrics(avg_metrics, feature_names, target_names)
            
            logging.info("Stacked ensemble model training and evaluation complete")
            return model, avg_metrics
            
        else:
            # Regular train/test split with chronological ordering
            # Sort by date if available
            if 'game_date' in data.columns:
                data = data.sort_values('game_date')
                # Re-create feature matrix after sorting
                X, y, feature_names = create_feature_matrix(data)
            
            # Take the last test_size% as test data (chronological split)
            split_idx = int(len(X) * (1 - test_size))
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            
            logging.info(f"Training set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")
            
            # Train stacked ensemble model
            model, _ = train_model(X_train, y_train, "stacked", None, low_resource_mode)
            
            # Evaluate the model
            metrics = evaluate_model(model, X_test, y_test, feature_names, target_names)
            
            # Save the model
            model_path = save_model(model, model_name=f"nba_stacked_{datetime.now().strftime('%Y%m%d')}.joblib")
            
            # Save the metrics
            metrics_path = save_metrics(metrics, feature_names, target_names)
            
            logging.info("Stacked ensemble model training and evaluation complete")
            return model, metrics
    
    elif train_separate_models:
        # Train separate models for each stat
        models = {}
        all_metrics = {}
        feature_importances = {}
        
        # For each target, use a different model if better performance is known
        target_models = {}
        
        # Train separate models with progress tracking
        with ProgressLogger(total=len(target_names), desc="Training separate models", unit="target") as progress:
            for i, target in enumerate(target_names):
                logging.info(f"Training model for {target}")
                
                # Extract the target column
                if isinstance(y, pd.DataFrame):
                    y_target = y[target]
                else:
                    y_target = y
                
                # Use time-based split to respect chronological order
                if use_cv:
                    # Create time series cross-validation with gap
                    tscv = TimeSeriesSplit(n_splits=5, gap=time_gap)
                    
                    # Initialize lists to store metrics
                    cv_scores = {
                        'mse': [],
                        'rmse': [],
                        'mae': [],
                        'r2': []
                    }
                    
                    # Select model type for this target if model_per_stat is provided
                    if model_per_stat and target in model_per_stat:
                        target_model_type = model_per_stat[target]
                        logging.info(f"Using {target_model_type} model for {target}")
                    else:
                        target_model_type = model_type
                    
                    # Store the model type used for this target
                    target_models[target] = target_model_type
                    
                    # Perform cross-validation
                    for train_idx, test_idx in tscv.split(X):
                        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                        y_train, y_test = y_target.iloc[train_idx], y_target.iloc[test_idx]
                        
                        # Tune hyperparameters if requested
                        if tune_hyperparams:
                            hyperparams = tune_model_hyperparameters(X_train, y_train, target_model_type, cv=3, 
                                                                    use_bayesian=use_bayesian_tuning)
                        else:
                            # Use default hyperparameters
                            hyperparams = None
                        
                        # Train the model
                        model, importance = train_model(X_train, y_train, target_model_type, hyperparams, low_resource_mode)
                        
                        # Evaluate on the test fold
                        y_pred = model.predict(X_test)
                        
                        # Calculate metrics
                        mse = mean_squared_error(y_test, y_pred)
                        mae = mean_absolute_error(y_test, y_pred)
                        r2 = r2_score(y_test, y_pred)
                        
                        # Store metrics
                        cv_scores['mse'].append(mse)
                        cv_scores['rmse'].append(np.sqrt(mse))
                        cv_scores['mae'].append(mae)
                        cv_scores['r2'].append(r2)
                    
                    # Calculate average scores
                    avg_metrics = {
                        'mse': np.mean(cv_scores['mse']),
                        'rmse': np.mean(cv_scores['rmse']),
                        'mae': np.mean(cv_scores['mae']),
                        'r2': np.mean(cv_scores['r2'])
                    }
                    
                    # Train final model on all data
                    model, importance = train_model(X, y_target, target_model_type, hyperparams, low_resource_mode)
                    feature_importances[target] = importance
                    
                    # Store the model and metrics
                    models[target] = model
                    all_metrics[target] = avg_metrics
                else:
                    # Sort by date if available
                    if 'game_date' in data.columns:
                        data = data.sort_values('game_date')
                        # Re-create feature matrix after sorting
                        X, y, feature_names = create_feature_matrix(data)
                        # Re-extract the target column
                        if isinstance(y, pd.DataFrame):
                            y_target = y[target]
                        else:
                            y_target = y
                    
                    # Take the last test_size% as test data (chronological split)
                    split_idx = int(len(X) * (1 - test_size))
                    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
                    y_train, y_test = y_target.iloc[:split_idx], y_target.iloc[split_idx:]
                    
                    # Select model type for this target if model_per_stat is provided
                    if model_per_stat and target in model_per_stat:
                        target_model_type = model_per_stat[target]
                        logging.info(f"Using {target_model_type} model for {target}")
                    else:
                        target_model_type = model_type
                    
                    # Store the model type used for this target
                    target_models[target] = target_model_type
                    
                    # Tune hyperparameters if requested
                    if tune_hyperparams:
                        hyperparams = tune_model_hyperparameters(X_train, y_train, target_model_type, 
                                                               use_bayesian=use_bayesian_tuning)
                    else:
                        # Use default hyperparameters
                        hyperparams = None
                    
                    # Train the model
                    model, importance = train_model(X_train, y_train, target_model_type, hyperparams, low_resource_mode)
                    feature_importances[target] = importance
                    
                    # Evaluate the model
                    y_pred = model.predict(X_test)
                    
                    # Calculate metrics
                    mse = mean_squared_error(y_test, y_pred)
                    mae = mean_absolute_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    
                    metrics = {
                        'mse': mse,
                        'rmse': np.sqrt(mse),
                        'mae': mae,
                        'r2': r2
                    }
                    
                    # Store the model and metrics
                    models[target] = model
                    all_metrics[target] = metrics
                
                # Log the R² for this target
                logging.info(f"R² for {target}: {all_metrics[target]['r2']:.4f} (using {target_models[target]})")
                
                # Update progress
                progress.update(1)
        
        # Calculate overall metrics (average across targets)
        overall_metrics = {
            'mse': np.mean([m['mse'] for m in all_metrics.values()]),
            'rmse': np.mean([m['rmse'] for m in all_metrics.values()]),
            'mae': np.mean([m['mae'] for m in all_metrics.values()]),
            'r2': np.mean([m['r2'] for m in all_metrics.values()])
        }
        
        # Add overall metrics
        all_metrics['overall'] = overall_metrics
        
        # Save models and metrics
        for target, model in models.items():
            model_type_used = target_models.get(target, model_type)
            model_path = save_model(model, model_name=f"nba_{model_type_used}_{target}_{datetime.now().strftime('%Y%m%d')}.joblib")
            logging.info(f"Saved {target} model to {model_path}")
        
        # Save all metrics
        metrics_path = save_metrics(all_metrics, feature_names, target_names)
        
        # Analyze feature importance if available
        if any(importance is not None for importance in feature_importances.values()):
            output_dir = "/Users/lukesmac/Projects/nbaModel/models"
            importance_results = {}
            
            for target, importance in feature_importances.items():
                if importance is not None:
                    # Create DataFrame with feature names and importance
                    importance_df = pd.DataFrame({
                        'feature': feature_names,
                        'importance': importance
                    })
                    importance_df = importance_df.sort_values('importance', ascending=False)
                    importance_results[target] = importance_df.head(15).to_dict('records')
            
            # Save feature importance
            current_date = datetime.now().strftime("%Y%m%d")
            importance_path = os.path.join(output_dir, f"feature_importance_{current_date}.json")
            
            try:
                import json
                with open(importance_path, 'w') as f:
                    json.dump(importance_results, f, indent=4)
                logging.info(f"Feature importance saved to {importance_path}")
            except Exception as e:
                logging.error(f"Error saving feature importance: {str(e)}")
        
        logging.info("Model training and evaluation complete for all targets")
        return models, all_metrics
    else:
        # Train a single multi-output model
        
        if use_cv:
            # Create time series cross-validation with gap
            tscv = TimeSeriesSplit(n_splits=5, gap=time_gap)
            
            # Initialize lists to store metrics
            cv_scores = {
                'overall': {'mse': [], 'rmse': [], 'mae': [], 'r2': []}
            }
            
            # Initialize target-specific metrics
            for target in target_names:
                cv_scores[target] = {'mse': [], 'rmse': [], 'mae': [], 'r2': []}
            
            # Perform cross-validation with progress tracking
            with ProgressLogger(total=len(list(tscv.split(X))), desc="Cross-validation", unit="fold") as progress:
                for train_idx, test_idx in tscv.split(X):
                    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                    
                    # Tune hyperparameters if requested
                    if tune_hyperparams:
                        hyperparams = tune_model_hyperparameters(X_train, y_train, model_type, cv=3, 
                                                               use_bayesian=use_bayesian_tuning)
                    else:
                        # Use default hyperparameters
                        hyperparams = None
                    
                    # Train the model
                    model, _ = train_model(X_train, y_train, model_type, hyperparams, low_resource_mode)
                    
                    # Evaluate on the test fold
                    y_pred = model.predict(X_test)
                    
                    # Calculate overall metrics
                    overall_mse = mean_squared_error(y_test, y_pred)
                    overall_mae = mean_absolute_error(y_test, y_pred)
                    overall_r2 = r2_score(y_test, y_pred)
                    
                    # Store overall metrics
                    cv_scores['overall']['mse'].append(overall_mse)
                    cv_scores['overall']['rmse'].append(np.sqrt(overall_mse))
                    cv_scores['overall']['mae'].append(overall_mae)
                    cv_scores['overall']['r2'].append(overall_r2)
                    
                    # Calculate and store target-specific metrics
                    for i, target in enumerate(target_names):
                        y_test_target = y_test.iloc[:, i]
                        y_pred_target = y_pred[:, i]
                        
                        mse = mean_squared_error(y_test_target, y_pred_target)
                        mae = mean_absolute_error(y_test_target, y_pred_target)
                        r2 = r2_score(y_test_target, y_pred_target)
                        
                        cv_scores[target]['mse'].append(mse)
                        cv_scores[target]['rmse'].append(np.sqrt(mse))
                        cv_scores[target]['mae'].append(mae)
                        cv_scores[target]['r2'].append(r2)
                    
                    progress.update(1)
            
            # Calculate average scores
            avg_metrics = {}
            for target, scores in cv_scores.items():
                avg_metrics[target] = {
                    'mse': np.mean(scores['mse']),
                    'rmse': np.mean(scores['rmse']),
                    'mae': np.mean(scores['mae']),
                    'r2': np.mean(scores['r2'])
                }
                
                # Print R² scores for each target
                logging.info(f"R² for {target}: {avg_metrics[target]['r2']:.4f}")
            
            # Train final model on all data
            model, importance = train_model(X, y, model_type, hyperparams, low_resource_mode)
            
            # Save the model and metrics
            model_path = save_model(model)
            metrics_path = save_metrics(avg_metrics, feature_names, target_names)
            
            # Analyze feature importance
            importance_analysis = analyze_feature_importance(model, X, y, feature_names, 
                                                            output_dir="/Users/lukesmac/Projects/nbaModel/models",
                                                            low_resource_mode=low_resource_mode)
            
            logging.info("Model training and evaluation complete")
            return model, avg_metrics
        else:
            # Regular train/test split with chronological ordering
            # Sort by date if available
            if 'game_date' in data.columns:
                data = data.sort_values('game_date')
                # Re-create feature matrix after sorting
                X, y, feature_names = create_feature_matrix(data)
            
            # Take the last test_size% as test data (chronological split)
            split_idx = int(len(X) * (1 - test_size))
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            
            logging.info(f"Training set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")
            
            # Tune hyperparameters if requested
            if tune_hyperparams:
                hyperparams = tune_model_hyperparameters(X_train, y_train, model_type, 
                                                       use_bayesian=use_bayesian_tuning)
            else:
                # Use default hyperparameters
                hyperparams = None
            
            # Train the model
            model, importance = train_model(X_train, y_train, model_type, hyperparams, low_resource_mode)
            
            # Evaluate the model
            metrics = evaluate_model(model, X_test, y_test, feature_names, target_names)
            
            # Save the model
            model_path = save_model(model)
            
            # Save the metrics
            metrics_path = save_metrics(metrics, feature_names, target_names)
            
            # Analyze feature importance
            importance_analysis = analyze_feature_importance(model, X, y, feature_names, 
                                                            output_dir="/Users/lukesmac/Projects/nbaModel/models",
                                                            low_resource_mode=low_resource_mode)
            
            logging.info("Model training and evaluation complete")
            return model, metrics

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train and evaluate NBA player performance prediction model')
    parser.add_argument('--data-path', type=str, help='Path to the engineered data file')
    parser.add_argument('--tune-hyperparams', action='store_true', help='Tune hyperparameters using grid search')
    parser.add_argument('--test-size', type=float, default=0.2, help='Proportion of data to use for testing')
    parser.add_argument('--random-state', type=int, default=42, help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Train and evaluate the model
    model, metrics = train_and_evaluate_model(
        data_path=args.data_path,
        tune_hyperparams=args.tune_hyperparams,
        test_size=args.test_size,
        random_state=args.random_state
    )