"""
Advanced feature importance visualization for XGBoost models
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import json
import joblib
import logging
from sklearn.inspection import permutation_importance
import shap
import xgboost as xgb

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_models_by_type(model_type='xgboost', models_dir=None, target=None):
    """
    Load XGBoost models from the models directory
    
    Args:
        model_type (str): Model type ('xgboost', 'random_forest', etc.)
        models_dir (str, optional): Path to models directory
        target (str, optional): Specific target to load (e.g., 'pts', 'ast')
        
    Returns:
        dict: Dictionary of loaded models by target
    """
    if models_dir is None:
        models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
    
    if not os.path.exists(models_dir):
        logging.error(f"Models directory not found: {models_dir}")
        return {}
    
    # Find model files based on type and target
    model_files = []
    
    if model_type == 'xgboost':
        # Look for XGBoost models
        if target:
            # Look for specific target
            xgb_files = glob.glob(os.path.join(models_dir, f"nba_{target}*xgboost*_model_*.joblib"))
            # Also check non-specific naming
            if not xgb_files:
                xgb_files = glob.glob(os.path.join(models_dir, f"nba_{target}_model_*.joblib"))
                # Filter later by checking model type
        else:
            # Look for all XGBoost models
            xgb_files = glob.glob(os.path.join(models_dir, "nba_*xgboost*_model_*.joblib"))
            if not xgb_files:
                # Try all model files and filter later
                xgb_files = glob.glob(os.path.join(models_dir, "nba_*_model_*.joblib"))
        
        model_files = xgb_files
    
    elif model_type == 'random_forest':
        # Look for RandomForest models
        if target:
            rf_files = glob.glob(os.path.join(models_dir, f"nba_{target}*rf*_model_*.joblib"))
            if not rf_files:
                rf_files = glob.glob(os.path.join(models_dir, f"nba_{target}_model_*.joblib"))
        else:
            rf_files = glob.glob(os.path.join(models_dir, "nba_*rf*_model_*.joblib"))
            if not rf_files:
                rf_files = glob.glob(os.path.join(models_dir, "nba_*_model_*.joblib"))
                
        model_files = rf_files
    
    else:
        # Generic model loading
        if target:
            model_files = glob.glob(os.path.join(models_dir, f"nba_{target}_model_*.joblib"))
        else:
            model_files = glob.glob(os.path.join(models_dir, "nba_*_model_*.joblib"))
    
    # Sort by name to get latest models first
    model_files.sort(reverse=True)
    
    # Load models
    loaded_models = {}
    
    for model_file in model_files:
        try:
            # Extract target from filename
            filename = os.path.basename(model_file)
            parts = filename.split('_')
            file_target = parts[1] if len(parts) > 1 else 'unknown'
            
            # Skip if we already have a more recent model for this target
            if file_target in loaded_models:
                continue
            
            # Load model
            model = joblib.load(model_file)
            
            # Check if this is the correct model type
            if model_type == 'xgboost' and hasattr(model, 'get_booster'):
                loaded_models[file_target] = {
                    'model': model,
                    'path': model_file
                }
            elif model_type == 'random_forest' and hasattr(model, 'estimators_'):
                loaded_models[file_target] = {
                    'model': model,
                    'path': model_file
                }
            elif model_type == 'all':
                # Accept any model type
                loaded_models[file_target] = {
                    'model': model,
                    'path': model_file
                }
            
        except Exception as e:
            logging.error(f"Error loading model {model_file}: {str(e)}")
    
    return loaded_models

def load_model_metrics(models_dir=None, target=None):
    """
    Load model metrics from the models directory
    
    Args:
        models_dir (str, optional): Path to models directory
        target (str, optional): Specific target to load metrics for
        
    Returns:
        dict: Dictionary of metrics by target
    """
    if models_dir is None:
        models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
    
    if not os.path.exists(models_dir):
        logging.error(f"Models directory not found: {models_dir}")
        return {}
    
    # Find metrics files
    metrics_files = []
    
    if target:
        metrics_files = glob.glob(os.path.join(models_dir, f"nba_{target}*_metrics_*.json"))
    else:
        metrics_files = glob.glob(os.path.join(models_dir, "nba_*_metrics_*.json"))
    
    # Sort by name to get latest metrics first
    metrics_files.sort(reverse=True)
    
    # Load metrics
    loaded_metrics = {}
    
    for metrics_file in metrics_files:
        try:
            # Extract target from filename
            filename = os.path.basename(metrics_file)
            parts = filename.split('_')
            file_target = parts[1] if len(parts) > 1 else 'unknown'
            
            # Skip if we already have more recent metrics for this target
            if file_target in loaded_metrics:
                continue
            
            # Load metrics
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            
            loaded_metrics[file_target] = {
                'metrics': metrics,
                'path': metrics_file
            }
            
        except Exception as e:
            logging.error(f"Error loading metrics {metrics_file}: {str(e)}")
    
    return loaded_metrics

def load_feature_importance(models_dir=None, target=None):
    """
    Load feature importance from the models directory
    
    Args:
        models_dir (str, optional): Path to models directory
        target (str, optional): Specific target to load feature importance for
        
    Returns:
        dict: Dictionary of feature importance by target
    """
    if models_dir is None:
        models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
    
    if not os.path.exists(models_dir):
        logging.error(f"Models directory not found: {models_dir}")
        return {}
    
    # Find feature importance files
    importance_files = []
    
    if target:
        importance_files = glob.glob(os.path.join(models_dir, f"feature_importance_{target}_*.json"))
    else:
        importance_files = glob.glob(os.path.join(models_dir, "feature_importance_*.json"))
    
    # Sort by name to get latest files first
    importance_files.sort(reverse=True)
    
    # Load feature importance
    loaded_importance = {}
    
    for importance_file in importance_files:
        try:
            # Extract target from filename
            filename = os.path.basename(importance_file)
            parts = filename.split('_')
            file_target = parts[2] if len(parts) > 2 else 'unknown'
            
            # Skip if we already have more recent importance for this target
            if file_target in loaded_importance:
                continue
            
            # Load importance
            with open(importance_file, 'r') as f:
                importance = json.load(f)
            
            loaded_importance[file_target] = {
                'importance': importance,
                'path': importance_file
            }
            
        except Exception as e:
            logging.error(f"Error loading feature importance {importance_file}: {str(e)}")
    
    return loaded_importance

def plot_feature_importance(feature_importance, target, output_dir=None, top_n=20, model_type='xgboost'):
    """
    Plot feature importance for a specific target
    
    Args:
        feature_importance (dict): Dictionary of feature importance values
        target (str): Target variable name
        output_dir (str, optional): Directory to save plot
        top_n (int): Number of top features to include
        model_type (str): Model type for labeling
        
    Returns:
        str: Path to saved plot
    """
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                'data', 'visualizations')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to DataFrame for easier plotting
    if isinstance(feature_importance, dict):
        importance_df = pd.DataFrame(list(feature_importance.items()), columns=['feature', 'importance'])
    elif isinstance(feature_importance, list):
        importance_df = pd.DataFrame(feature_importance, columns=['feature', 'importance'])
    else:
        logging.error(f"Unrecognized feature importance format: {type(feature_importance)}")
        return None
    
    # Sort by importance and get top features
    importance_df = importance_df.sort_values('importance', ascending=False).head(top_n)
    
    # Set up the style
    sns.set(style='whitegrid', font_scale=1.1)
    
    # Create plot
    plt.figure(figsize=(12, 10))
    bars = sns.barplot(x='importance', y='feature', data=importance_df, palette='viridis')
    
    # Add value labels to bars
    for i, v in enumerate(importance_df['importance']):
        bars.text(v + 0.002, i, f"{v:.4f}", va='center')
    
    # Set title and labels
    plt.title(f'Top {top_n} Features for {target.upper()} Prediction ({model_type.upper()})', fontsize=16)
    plt.xlabel('Feature Importance', fontsize=14)
    plt.ylabel('Feature', fontsize=14)
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(output_dir, f'feature_importance_{target}_{model_type}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Saved feature importance plot to {plot_path}")
    return plot_path

def calculate_xgboost_feature_importance(model, X_test, y_test, feature_names=None, importance_type='gain'):
    """
    Calculate XGBoost-specific feature importance
    
    Args:
        model: XGBoost model
        X_test: Test features
        y_test: Test targets
        feature_names (list, optional): Feature names
        importance_type (str): XGBoost importance type ('gain', 'weight', 'cover', 'total_gain', 'total_cover')
        
    Returns:
        dict: Dictionary of feature importance values
    """
    if not hasattr(model, 'get_booster'):
        logging.error("Model is not an XGBoost model")
        return {}
    
    # Get feature names if not provided
    if feature_names is None:
        if hasattr(model, 'feature_names'):
            feature_names = model.feature_names
        elif hasattr(X_test, 'columns'):
            feature_names = X_test.columns.tolist()
        else:
            # Create generic feature names
            feature_names = [f'f{i}' for i in range(X_test.shape[1])]
    
    # Get feature importance from XGBoost model
    booster = model.get_booster()
    importance = booster.get_score(importance_type=importance_type)
    
    # If some features are missing, fill with 0
    all_features = {}
    for i, name in enumerate(feature_names):
        all_features[name] = importance.get(name, 0)
    
    # Sort by importance
    sorted_importance = {k: v for k, v in sorted(all_features.items(), key=lambda item: item[1], reverse=True)}
    return sorted_importance

def calculate_shap_importance(model, X_test, feature_names=None):
    """
    Calculate SHAP feature importance
    
    Args:
        model: Trained model (XGBoost recommended)
        X_test: Test features
        feature_names (list, optional): Feature names
        
    Returns:
        dict: Dictionary of feature importance values based on SHAP
    """
    try:
        # Check if SHAP is usable with this model
        if not hasattr(model, 'predict'):
            logging.error("Model does not have predict method required for SHAP")
            return {}
        
        # Get feature names if not provided
        if feature_names is None:
            if hasattr(model, 'feature_names'):
                feature_names = model.feature_names
            elif hasattr(X_test, 'columns'):
                feature_names = X_test.columns.tolist()
            else:
                # Create generic feature names
                feature_names = [f'f{i}' for i in range(X_test.shape[1])]
        
        # For XGBoost models, use the optimized TreeExplainer
        if hasattr(model, 'get_booster'):
            explainer = shap.TreeExplainer(model)
        else:
            # For other models, use KernelExplainer with a sample of data
            # (use a subset of X_test to speed up computation)
            sample_size = min(100, X_test.shape[0])
            sample_indices = np.random.choice(X_test.shape[0], sample_size, replace=False)
            X_sample = X_test.iloc[sample_indices] if hasattr(X_test, 'iloc') else X_test[sample_indices]
            explainer = shap.KernelExplainer(model.predict, X_sample)
        
        # Calculate SHAP values (use a subset for large datasets)
        if X_test.shape[0] > 500:
            sample_indices = np.random.choice(X_test.shape[0], 500, replace=False)
            X_sample = X_test.iloc[sample_indices] if hasattr(X_test, 'iloc') else X_test[sample_indices]
            shap_values = explainer.shap_values(X_sample)
        else:
            shap_values = explainer.shap_values(X_test)
        
        # For multi-output models, take the first output
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        
        # Calculate mean absolute SHAP value for each feature
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        # Create dictionary of feature importance
        importance_dict = {}
        for i, name in enumerate(feature_names):
            importance_dict[name] = float(mean_abs_shap[i])
        
        # Sort by importance
        sorted_importance = {k: v for k, v in sorted(importance_dict.items(), key=lambda item: item[1], reverse=True)}
        return sorted_importance
    
    except Exception as e:
        logging.error(f"Error calculating SHAP importance: {str(e)}")
        return {}

def plot_shap_summary(model, X_test, output_dir=None, target='unknown', max_display=20):
    """
    Create SHAP summary plot
    
    Args:
        model: Trained model (XGBoost recommended)
        X_test: Test features
        output_dir (str, optional): Directory to save plot
        target (str): Target variable name
        max_display (int): Maximum number of features to show
        
    Returns:
        str: Path to saved plot
    """
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                'data', 'visualizations')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # For XGBoost models, use the optimized TreeExplainer
        if hasattr(model, 'get_booster'):
            explainer = shap.TreeExplainer(model)
        else:
            # For other models, use KernelExplainer with a sample of data
            sample_size = min(100, X_test.shape[0])
            sample_indices = np.random.choice(X_test.shape[0], sample_size, replace=False)
            X_sample = X_test.iloc[sample_indices] if hasattr(X_test, 'iloc') else X_test[sample_indices]
            explainer = shap.KernelExplainer(model.predict, X_sample)
        
        # Calculate SHAP values (use a subset for large datasets)
        if X_test.shape[0] > 500:
            sample_indices = np.random.choice(X_test.shape[0], 500, replace=False)
            X_sample = X_test.iloc[sample_indices] if hasattr(X_test, 'iloc') else X_test[sample_indices]
            shap_values = explainer.shap_values(X_sample)
            X_display = X_sample
        else:
            shap_values = explainer.shap_values(X_test)
            X_display = X_test
        
        # For multi-output models, take the first output
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        
        # Create summary plot
        plt.figure(figsize=(12, 10))
        shap.summary_plot(shap_values, X_display, max_display=max_display, show=False)
        plt.title(f'SHAP Feature Importance for {target.upper()}', fontsize=16)
        plt.tight_layout()
        
        # Save the plot
        plot_path = os.path.join(output_dir, f'shap_summary_{target}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Saved SHAP summary plot to {plot_path}")
        return plot_path
    
    except Exception as e:
        logging.error(f"Error creating SHAP summary plot: {str(e)}")
        return None

def create_feature_importance_visualizations(test_data_path=None, models_dir=None, output_dir=None, 
                                           model_type='xgboost', targets=None, top_n=20, use_shap=True):
    """
    Create feature importance visualizations for models
    
    Args:
        test_data_path (str, optional): Path to test data file
        models_dir (str, optional): Path to models directory
        output_dir (str, optional): Directory to save plots
        model_type (str): Model type to visualize
        targets (list, optional): List of targets to visualize
        top_n (int): Number of top features to include
        use_shap (bool): Whether to use SHAP for visualization
        
    Returns:
        list: Paths to saved plots
    """
    if models_dir is None:
        models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
    
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                'data', 'visualizations')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load models
    models = load_models_by_type(model_type=model_type, models_dir=models_dir, target=None)
    if not models:
        logging.error(f"No {model_type} models found")
        return []
    
    # If targets not specified, use all loaded models
    if targets is None:
        targets = list(models.keys())
    else:
        # Filter to only include targets that we have models for
        targets = [t for t in targets if t in models]
    
    if not targets:
        logging.error("No valid targets to visualize")
        return []
    
    # Load test data if path provided
    X_test = None
    y_test = None
    if test_data_path and os.path.exists(test_data_path):
        try:
            test_data = pd.read_csv(test_data_path)
            
            # Try to identify target columns using patterns
            target_pattern = '|'.join(targets)
            potential_target_cols = [col for col in test_data.columns if re.search(f"^({target_pattern})$", col, re.IGNORECASE)]
            
            if potential_target_cols:
                # Split data into features and targets
                y_test = test_data[potential_target_cols]
                X_test = test_data.drop(columns=potential_target_cols)
                logging.info(f"Loaded test data with {X_test.shape[1]} features and {len(potential_target_cols)} targets")
            else:
                # No clear target columns, just use first 80% columns as features
                feature_count = int(test_data.shape[1] * 0.8)
                X_test = test_data.iloc[:, :feature_count]
                logging.info(f"Loaded test data with {X_test.shape[1]} features (no clear targets found)")
        except Exception as e:
            logging.error(f"Error loading test data: {str(e)}")
            X_test = None
    
    # Load feature importance files
    importance_data = load_feature_importance(models_dir=models_dir)
    
    # Plot paths
    plot_paths = []
    
    # Create visualizations for each target
    for target in targets:
        if target not in models:
            logging.warning(f"No model found for target {target}, skipping")
            continue
        
        model_info = models[target]
        model = model_info['model']
        
        # Use existing feature importance if available
        if target in importance_data and 'importance' in importance_data[target]:
            feature_importance = importance_data[target]['importance']
            logging.info(f"Using existing feature importance for {target}")
            
            # Create standard plot
            plot_path = plot_feature_importance(
                feature_importance=feature_importance,
                target=target,
                output_dir=output_dir,
                top_n=top_n,
                model_type=model_type
            )
            
            if plot_path:
                plot_paths.append(plot_path)
        
        # Calculate feature importance if we have test data
        elif X_test is not None:
            logging.info(f"Calculating feature importance for {target}")
            feature_importance = calculate_xgboost_feature_importance(
                model=model,
                X_test=X_test,
                y_test=y_test[target] if y_test is not None and target in y_test else None,
                feature_names=X_test.columns if hasattr(X_test, 'columns') else None,
                importance_type='gain'
            )
            
            # Create standard plot
            plot_path = plot_feature_importance(
                feature_importance=feature_importance,
                target=target,
                output_dir=output_dir,
                top_n=top_n,
                model_type=model_type
            )
            
            if plot_path:
                plot_paths.append(plot_path)
            
            # Save feature importance to file
            importance_path = os.path.join(models_dir, f"feature_importance_{target}_{model_type}.json")
            try:
                with open(importance_path, 'w') as f:
                    json.dump(feature_importance, f, indent=2)
                logging.info(f"Saved feature importance to {importance_path}")
            except Exception as e:
                logging.error(f"Error saving feature importance: {str(e)}")
        
        # Create SHAP visualization if requested and we have test data
        if use_shap and X_test is not None:
            logging.info(f"Creating SHAP visualization for {target}")
            
            # Create SHAP summary plot
            shap_path = plot_shap_summary(
                model=model,
                X_test=X_test,
                output_dir=output_dir,
                target=target,
                max_display=top_n
            )
            
            if shap_path:
                plot_paths.append(shap_path)
            
            # Calculate and save SHAP feature importance
            if X_test is not None:
                shap_importance = calculate_shap_importance(
                    model=model,
                    X_test=X_test,
                    feature_names=X_test.columns if hasattr(X_test, 'columns') else None
                )
                
                # Save SHAP importance to file
                shap_importance_path = os.path.join(models_dir, f"feature_importance_{target}_shap.json")
                try:
                    with open(shap_importance_path, 'w') as f:
                        json.dump(shap_importance, f, indent=2)
                    logging.info(f"Saved SHAP feature importance to {shap_importance_path}")
                except Exception as e:
                    logging.error(f"Error saving SHAP feature importance: {str(e)}")
    
    return plot_paths

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create feature importance visualizations for models")
    
    parser.add_argument("--test-data", type=str, help="Path to test data file")
    parser.add_argument("--models-dir", type=str, help="Path to models directory")
    parser.add_argument("--output-dir", type=str, help="Directory to save plots")
    parser.add_argument("--model-type", type=str, default="xgboost", 
                        choices=["xgboost", "random_forest", "all"],
                        help="Model type to visualize")
    parser.add_argument("--targets", type=str, nargs="+", help="Targets to visualize")
    parser.add_argument("--top-n", type=int, default=20, help="Number of top features to include")
    parser.add_argument("--no-shap", action="store_true", help="Don't use SHAP for visualization")
    
    args = parser.parse_args()
    
    # Create visualizations
    plot_paths = create_feature_importance_visualizations(
        test_data_path=args.test_data,
        models_dir=args.models_dir,
        output_dir=args.output_dir,
        model_type=args.model_type,
        targets=args.targets,
        top_n=args.top_n,
        use_shap=not args.no_shap
    )
    
    print(f"Created {len(plot_paths)} visualizations")
    for path in plot_paths:
        print(f"  - {path}")