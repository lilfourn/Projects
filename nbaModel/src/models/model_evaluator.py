#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Evaluation Module

This module provides functions to evaluate NBA prediction models,
including metrics calculation, cross-validation, and prediction analysis.
"""

import os
import sys
import logging
import json
import glob
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    explained_variance_score, mean_absolute_percentage_error
)
from sklearn.model_selection import cross_val_score, KFold
import matplotlib.pyplot as plt
import seaborn as sns

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Import from src modules
try:
    from src.utils.config import MODELS_DIR, CURRENT_DATE
    from src.visualization.model_visualizer import (
        visualize_feature_importance,
        visualize_model_metrics,
        visualize_prediction_analysis
    )
except ImportError:
    # Try direct import for when running from models directory
    try:
        from utils.config import MODELS_DIR, CURRENT_DATE
        from visualization.model_visualizer import (
            visualize_feature_importance,
            visualize_model_metrics,
            visualize_prediction_analysis
        )
    except ImportError:
        logging.error("Failed to import required modules. Make sure you're running from the project root.")
        # Use default values
        MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models")
        CURRENT_DATE = datetime.now().strftime("%Y%m%d")
        # Set visualization functions to None
        visualize_feature_importance = None
        visualize_model_metrics = None
        visualize_prediction_analysis = None

# Function to get target-specific directory for model files
def get_target_model_dir(target_name):
    """
    Get the directory for a specific target's model files
    
    Args:
        target_name (str): Name of the target variable (e.g., pts, reb, ast)
        
    Returns:
        str: Path to the target-specific model directory
    """
    if target_name is None:
        return MODELS_DIR
    
    # Create the target directory
    target_dir = os.path.join(MODELS_DIR, target_name.lower())
    os.makedirs(target_dir, exist_ok=True)
    
    return target_dir

def evaluate_model(model, X, y, target_name, cv=5, visualize=True):
    """
    Evaluate a trained model with various metrics and cross-validation
    
    Args:
        model: Trained model object with predict method
        X (DataFrame): Feature matrix
        y (Series): Target values
        target_name (str): Name of the target variable
        cv (int): Number of cross-validation folds
        visualize (bool): Whether to generate visualizations
        
    Returns:
        dict: Dictionary of evaluation metrics
    """
    # Make predictions
    y_pred = model.predict(X)
    
    # Calculate metrics
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    explained_var = explained_variance_score(y, y_pred)
    
    # Calculate MAPE only if y doesn't contain zeros
    if not np.any(y == 0):
        mape = mean_absolute_percentage_error(y, y_pred)
    else:
        # Calculate custom MAPE excluding zeros
        mask = y != 0
        if np.sum(mask) > 0:
            mape = np.mean(np.abs((y[mask] - y_pred[mask]) / y[mask])) * 100
        else:
            mape = np.nan
    
    # Perform cross-validation
    cv_scores = cross_val_score(
        model, X, y, cv=KFold(n_splits=cv, shuffle=True, random_state=42),
        scoring='r2'
    )
    
    cv_r2_mean = np.mean(cv_scores)
    cv_r2_std = np.std(cv_scores)
    
    # Compile metrics
    metrics = {
        'r2': float(r2),
        'mae': float(mae),
        'rmse': float(rmse),
        'explained_variance': float(explained_var),
        'mape': float(mape) if not np.isnan(mape) else None,
        'cv_r2_mean': float(cv_r2_mean),
        'cv_r2_std': float(cv_r2_std),
        'cv_r2_scores': cv_scores.tolist(),
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Log the metrics
    logging.info(f"Model evaluation for {target_name}:")
    logging.info(f"  R² score: {r2:.4f}")
    logging.info(f"  Mean absolute error: {mae:.4f}")
    logging.info(f"  Root mean squared error: {rmse:.4f}")
    if metrics['mape'] is not None:
        logging.info(f"  Mean absolute percentage error: {mape:.4f}%")
    logging.info(f"  Cross-validation R² score: {cv_r2_mean:.4f} ± {cv_r2_std:.4f}")
    
    # Generate visualizations if requested
    if visualize and visualize_prediction_analysis is not None:
        try:
            date_str = CURRENT_DATE
            visualize_prediction_analysis(
                target_name=target_name,
                actual=y,
                predicted=y_pred,
                date_str=date_str,
                save_fig=True,
                show_fig=False
            )
        except Exception as e:
            logging.error(f"Error generating prediction analysis visualization: {str(e)}")
    
    return metrics

def save_model_metrics(metrics, target_name, model_type=None, date_str=None):
    """
    Save model evaluation metrics to a JSON file
    
    Args:
        metrics (dict): Dictionary of evaluation metrics
        target_name (str): Name of the target variable
        model_type (str, optional): Type of model (e.g., random_forest, xgboost)
        date_str (str, optional): Date string for the filename. If None, will use current date.
        
    Returns:
        str: Path to the saved metrics file
    """
    if date_str is None:
        date_str = CURRENT_DATE
    
    # Get target-specific directory
    target_dir = get_target_model_dir(target_name)
    
    # Create a professional filename
    if model_type:
        filename = f"nba_{target_name}_{model_type}_metrics_{date_str}.json"
    else:
        filename = f"nba_{target_name}_metrics_{date_str}.json"
    
    # Full path to the metrics file
    metrics_file = os.path.join(target_dir, filename)
    
    # Save metrics to JSON file
    try:
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=4)
        logging.info(f"Saved model metrics to {metrics_file}")
        return metrics_file
    except Exception as e:
        logging.error(f"Error saving metrics to {metrics_file}: {str(e)}")
        return None

def save_feature_importance(feature_importance_df, target_name, model_type=None, date_str=None):
    """
    Save feature importance data to a JSON file
    
    Args:
        feature_importance_df (pandas.DataFrame): DataFrame with 'feature' and 'importance' columns
        target_name (str): Name of the target variable
        model_type (str, optional): Type of model (e.g., random_forest, xgboost)
        date_str (str, optional): Date string for the filename. If None, will use current date.
        
    Returns:
        str: Path to the saved feature importance file
    """
    # Use current date if not provided
    if date_str is None:
        date_str = CURRENT_DATE
    
    # Get target-specific directory
    target_dir = get_target_model_dir(target_name)
    
    # Create a professional filename
    if model_type:
        filename = f"nba_{target_name}_{model_type}_feature_importance_{date_str}.json"
    else:
        filename = f"nba_{target_name}_feature_importance_{date_str}.json"
    
    # Full path to the feature importance file
    importance_file = os.path.join(target_dir, filename)
    
    # Convert DataFrame to dictionary for JSON serialization
    importance_dict = {row['feature']: float(row['importance']) 
                      for _, row in feature_importance_df.iterrows()}
    
    # Save feature importance to file
    try:
        with open(importance_file, 'w') as f:
            json.dump(importance_dict, f, indent=4)
        logging.info(f"Saved feature importance to {importance_file}")
        return importance_file
    except Exception as e:
        logging.error(f"Error saving feature importance to {importance_file}: {str(e)}")
        return None

def compare_models(target_name, models_info, X, y, visualize=True):
    """
    Compare multiple models for the same target variable
    
    Args:
        target_name (str): Name of the target variable
        models_info (list): List of dictionaries with keys 'name', 'model'
        X (DataFrame): Feature matrix
        y (Series): Target values
        visualize (bool): Whether to generate visualizations
        
    Returns:
        dict: Dictionary with model names as keys and metrics as values
    """
    results = {}
    
    for model_info in models_info:
        model_name = model_info['name']
        model = model_info['model']
        
        logging.info(f"Evaluating {model_name} model for {target_name}")
        
        # Evaluate the model
        metrics = evaluate_model(
            model=model,
            X=X,
            y=y,
            target_name=f"{target_name}_{model_name}",
            visualize=False  # Don't visualize individual models
        )
        
        # Store the results
        results[model_name] = metrics
    
    # Create a comparison table
    comparison = pd.DataFrame({
        'Model': [info['name'] for info in models_info],
        'R²': [results[info['name']]['r2'] for info in models_info],
        'MAE': [results[info['name']]['mae'] for info in models_info],
        'RMSE': [results[info['name']]['rmse'] for info in models_info],
        'CV R²': [results[info['name']]['cv_r2_mean'] for info in models_info]
    })
    
    # Sort by R² (descending)
    comparison = comparison.sort_values('R²', ascending=False)
    
    # Log the comparison
    logging.info("\nModel Comparison:")
    logging.info("=" * 60)
    logging.info(f"{comparison.to_string(index=False)}")
    logging.info("=" * 60)
    
    # Save the comparison
    date_str = CURRENT_DATE
    filename = f"model_comparison_{target_name}_{date_str}.json"
    file_path = os.path.join(get_target_model_dir(target_name), filename)
    
    try:
        with open(file_path, 'w') as f:
            json.dump(results, f, indent=2)
        logging.info(f"Saved model comparison to {file_path}")
    except Exception as e:
        logging.error(f"Error saving model comparison: {str(e)}")
    
    # Generate visualization if requested
    if visualize and visualize_model_metrics is not None:
        try:
            # Create a custom metrics structure for visualization
            metrics_data = {}
            for model_name, metrics in results.items():
                metrics_data[f"{target_name}_{model_name}"] = metrics
            
            # Save temporary metrics files for visualization
            for model_target, metrics in metrics_data.items():
                temp_file = os.path.join(get_target_model_dir(target_name), f"nba_{model_target}_metrics_{date_str}.json")
                with open(temp_file, 'w') as f:
                    json.dump(metrics, f, indent=2)
            
            # Generate visualization
            visualize_model_metrics(
                target_name=None,  # Visualize all targets (which are model variants)
                date_str=date_str,
                save_fig=True,
                show_fig=False
            )
            
            # Clean up temporary files
            for model_target in metrics_data.keys():
                temp_file = os.path.join(get_target_model_dir(target_name), f"nba_{model_target}_metrics_{date_str}.json")
                if os.path.exists(temp_file):
                    os.remove(temp_file)
        except Exception as e:
            logging.error(f"Error generating model comparison visualization: {str(e)}")
    
    return results

def generate_model_report(target_name, date_str=None, output_format='markdown'):
    """
    Generate a comprehensive report for a model
    
    Args:
        target_name (str): Name of the target variable
        date_str (str, optional): Date string for the model files. If None, will use the latest.
        output_format (str): Format of the report ('markdown' or 'html')
        
    Returns:
        str: Report content in the specified format
    """
    # Find the model files
    if date_str is None:
        # Find the latest model file for this target
        model_files = glob.glob(os.path.join(get_target_model_dir(target_name), f"nba_{target_name}_model_*.joblib"))
        if not model_files:
            logging.error(f"No model files found for target: {target_name}")
            return None
        
        # Sort by modification time (newest first)
        model_files.sort(key=os.path.getmtime, reverse=True)
        model_file = model_files[0]
        # Extract date from filename
        date_str = os.path.basename(model_file).split('_')[-1].split('.')[0]
    
    # Load the metrics
    metrics_file = os.path.join(get_target_model_dir(target_name), f"nba_{target_name}_metrics_{date_str}.json")
    if not os.path.exists(metrics_file):
        logging.error(f"Metrics file not found: {metrics_file}")
        return None
    
    try:
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
    except Exception as e:
        logging.error(f"Error loading metrics: {str(e)}")
        return None
    
    # Load feature importance
    importance_file = os.path.join(get_target_model_dir(target_name), f"nba_{target_name}_feature_importance_{date_str}.json")
    if not os.path.exists(importance_file):
        logging.warning(f"Feature importance file not found: {importance_file}")
        importance_data = None
    else:
        try:
            with open(importance_file, 'r') as f:
                importance_data = json.load(f)
        except Exception as e:
            logging.error(f"Error loading feature importance: {str(e)}")
            importance_data = None
    
    # Generate the report
    if output_format == 'markdown':
        report = generate_markdown_report(target_name, metrics, importance_data, date_str)
    elif output_format == 'html':
        report = generate_html_report(target_name, metrics, importance_data, date_str)
    else:
        logging.error(f"Unsupported output format: {output_format}")
        return None
    
    # Save the report
    report_dir = os.path.join(os.path.dirname(MODELS_DIR), "reports")
    os.makedirs(report_dir, exist_ok=True)
    
    extension = 'md' if output_format == 'markdown' else 'html'
    report_file = os.path.join(report_dir, f"model_report_{target_name}_{date_str}.{extension}")
    
    try:
        with open(report_file, 'w') as f:
            f.write(report)
        logging.info(f"Saved model report to {report_file}")
        return report_file
    except Exception as e:
        logging.error(f"Error saving report: {str(e)}")
        return None

def generate_markdown_report(target_name, metrics, importance_data, date_str):
    """Generate a markdown report for a model"""
    # Format date for display
    display_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
    
    report = f"""# NBA Prediction Model Report: {target_name.upper()}

## Model Overview
- **Target Variable**: {target_name.upper()}
- **Date**: {display_date}
- **Timestamp**: {metrics.get('timestamp', 'N/A')}

## Performance Metrics

| Metric | Value |
|--------|-------|
| R² Score | {metrics.get('r2', 'N/A'):.4f} |
| Mean Absolute Error (MAE) | {metrics.get('mae', 'N/A'):.4f} |
| Root Mean Squared Error (RMSE) | {metrics.get('rmse', 'N/A'):.4f} |
"""
    
    if metrics.get('mape') is not None:
        report += f"| Mean Absolute Percentage Error (MAPE) | {metrics.get('mape', 'N/A'):.4f}% |\n"
    
    report += f"""| Explained Variance | {metrics.get('explained_variance', 'N/A'):.4f} |
| Cross-Validation R² | {metrics.get('cv_r2_mean', 'N/A'):.4f} ± {metrics.get('cv_r2_std', 'N/A'):.4f} |

## Cross-Validation Results

"""
    
    cv_scores = metrics.get('cv_r2_scores', [])
    if cv_scores:
        report += "| Fold | R² Score |\n|------|----------|\n"
        for i, score in enumerate(cv_scores):
            report += f"| {i+1} | {score:.4f} |\n"
    else:
        report += "No cross-validation scores available.\n"
    
    report += "\n## Feature Importance\n\n"
    
    if importance_data:
        report += "| Feature | Importance |\n|---------|------------|\n"
        
        # Show top 20 features
        top_n = min(20, len(importance_data))
        for i, (feature, importance) in enumerate(list(importance_data.items())[:top_n]):
            report += f"| {feature} | {importance:.4f} |\n"
    else:
        report += "No feature importance data available.\n"
    
    report += f"""
## Visualizations

The following visualizations have been generated for this model:

1. Feature Importance: `visualizations/feature_importance_{target_name}_top20_{date_str}.png`
2. Model Metrics: `visualizations/model_metrics_{target_name}_{date_str}.png`
3. Prediction Analysis: `visualizations/prediction_analysis_{target_name}_{date_str}.png`

Interactive versions are also available as HTML files in the visualizations directory.

## Model Files

- Model: `models/{target_name}/nba_{target_name}_model_{date_str}.joblib`
- Metrics: `models/{target_name}/nba_{target_name}_metrics_{date_str}.json`
- Feature Importance: `models/{target_name}/nba_{target_name}_feature_importance_{date_str}.json`

---

*Report generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
"""
    
    return report

def generate_html_report(target_name, metrics, importance_data, date_str):
    """Generate an HTML report for a model"""
    # Format date for display
    display_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
    
    # Start HTML document
    report = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NBA Prediction Model Report: {target_name.upper()}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}
        h1, h2, h3 {{
            color: #2c3e50;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }}
        th {{
            background-color: #f2f2f2;
        }}
        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        .metric-value {{
            font-weight: bold;
        }}
        .visualization {{
            margin: 20px 0;
            text-align: center;
        }}
        .visualization img {{
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
        }}
        footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #eee;
            font-size: 0.9em;
            color: #777;
        }}
    </style>
</head>
<body>
    <h1>NBA Prediction Model Report: {target_name.upper()}</h1>
    
    <h2>Model Overview</h2>
    <table>
        <tr>
            <th>Target Variable</th>
            <td>{target_name.upper()}</td>
        </tr>
        <tr>
            <th>Date</th>
            <td>{display_date}</td>
        </tr>
        <tr>
            <th>Timestamp</th>
            <td>{metrics.get('timestamp', 'N/A')}</td>
        </tr>
    </table>
    
    <h2>Performance Metrics</h2>
    <table>
        <tr>
            <th>Metric</th>
            <th>Value</th>
        </tr>
        <tr>
            <th>R² Score</th>
            <td class="metric-value">{metrics.get('r2', 'N/A'):.4f}</td>
        </tr>
        <tr>
            <th>Mean Absolute Error (MAE)</th>
            <td class="metric-value">{metrics.get('mae', 'N/A'):.4f}</td>
        </tr>
        <tr>
            <th>Root Mean Squared Error (RMSE)</th>
            <td class="metric-value">{metrics.get('rmse', 'N/A'):.4f}</td>
        </tr>
"""
    
    if metrics.get('mape') is not None:
        report += f"""        <tr>
            <th>Mean Absolute Percentage Error (MAPE)</th>
            <td class="metric-value">{metrics.get('mape', 'N/A'):.4f}%</td>
        </tr>
"""
    
    report += f"""        <tr>
            <th>Explained Variance</th>
            <td class="metric-value">{metrics.get('explained_variance', 'N/A'):.4f}</td>
        </tr>
        <tr>
            <th>Cross-Validation R²</th>
            <td class="metric-value">{metrics.get('cv_r2_mean', 'N/A'):.4f} ± {metrics.get('cv_r2_std', 'N/A'):.4f}</td>
        </tr>
    </table>
    
    <h2>Cross-Validation Results</h2>
"""
    
    cv_scores = metrics.get('cv_r2_scores', [])
    if cv_scores:
        report += """    <table>
        <tr>
            <th>Fold</th>
            <th>R² Score</th>
        </tr>
"""
        for i, score in enumerate(cv_scores):
            report += f"""        <tr>
            <td>{i+1}</td>
            <td>{score:.4f}</td>
        </tr>
"""
        report += "    </table>\n"
    else:
        report += "    <p>No cross-validation scores available.</p>\n"
    
    report += "    <h2>Feature Importance</h2>\n"
    
    if importance_data:
        report += """    <table>
        <tr>
            <th>Feature</th>
            <th>Importance</th>
        </tr>
"""
        # Show top 20 features
        top_n = min(20, len(importance_data))
        for i, (feature, importance) in enumerate(list(importance_data.items())[:top_n]):
            report += f"""        <tr>
            <td>{feature}</td>
            <td>{importance:.4f}</td>
        </tr>
"""
        report += "    </table>\n"
    else:
        report += "    <p>No feature importance data available.</p>\n"
    
    report += f"""
    <h2>Visualizations</h2>
    
    <p>The following visualizations have been generated for this model:</p>
    
    <div class="visualization">
        <h3>Feature Importance</h3>
        <img src="../visualizations/feature_importance_{target_name}_top20_{date_str}.png" alt="Feature Importance">
        <p><a href="../visualizations/feature_importance_{target_name}_top20_{date_str}.html" target="_blank">View Interactive Version</a></p>
    </div>
    
    <div class="visualization">
        <h3>Model Metrics</h3>
        <img src="../visualizations/model_metrics_{target_name}_{date_str}.png" alt="Model Metrics">
        <p><a href="../visualizations/model_metrics_{target_name}_{date_str}.html" target="_blank">View Interactive Version</a></p>
    </div>
    
    <div class="visualization">
        <h3>Prediction Analysis</h3>
        <img src="../visualizations/prediction_analysis_{target_name}_{date_str}.png" alt="Prediction Analysis">
        <p><a href="../visualizations/prediction_analysis_{target_name}_{date_str}.html" target="_blank">View Interactive Version</a></p>
    </div>
    
    <h2>Model Files</h2>
    
    <ul>
        <li>Model: <code>models/{target_name}/nba_{target_name}_model_{date_str}.joblib</code></li>
        <li>Metrics: <code>models/{target_name}/nba_{target_name}_metrics_{date_str}.json</code></li>
        <li>Feature Importance: <code>models/{target_name}/nba_{target_name}_feature_importance_{date_str}.json</code></li>
    </ul>
    
    <footer>
        <p>Report generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    </footer>
</body>
</html>
"""
    
    return report

def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate model evaluation reports')
    parser.add_argument('--target', type=str, required=True,
                        help='Target variable (e.g., pts, reb, ast)')
    parser.add_argument('--date', type=str, default=None,
                        help='Date string for model files (format: YYYYMMDD)')
    parser.add_argument('--format', type=str, choices=['markdown', 'html'], default='markdown',
                        help='Output format for the report')
    
    args = parser.parse_args()
    
    # Generate the report
    report_file = generate_model_report(
        target_name=args.target,
        date_str=args.date,
        output_format=args.format
    )
    
    if report_file:
        logging.info(f"Report generated successfully: {report_file}")
    else:
        logging.error("Failed to generate report")

if __name__ == "__main__":
    main()
