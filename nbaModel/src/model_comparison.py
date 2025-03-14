"""
Utilities for comparing model performance across different models
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter
import logging
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_model_metrics(models_dir=None):
    """
    Load metrics for all models in the models directory
    
    Args:
        models_dir (str, optional): Path to models directory
        
    Returns:
        dict: Dictionary of model metrics by model type and target
    """
    if models_dir is None:
        models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
    
    if not os.path.exists(models_dir):
        logging.error(f"Models directory not found: {models_dir}")
        return {}
    
    # Scan for all metrics files
    metrics_files = []
    for file in os.listdir(models_dir):
        if file.endswith('.json') and '_metrics_' in file:
            metrics_files.append(os.path.join(models_dir, file))
    
    # Sort by filename (which includes date)
    metrics_files.sort()
    
    # Process each metrics file
    all_metrics = defaultdict(dict)
    
    for metrics_file in metrics_files:
        try:
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            
            # Extract model details from filename
            filename = os.path.basename(metrics_file)
            parts = filename.replace('.json', '').split('_')
            
            # Determine model type
            model_type = 'random_forest'  # Default
            if 'xgboost' in filename:
                model_type = 'xgboost'
            elif any(tech in filename for tech in ['gb', 'gradientboosting']):
                model_type = 'gradient_boosting'
            elif 'lgbm' in filename or 'lightgbm' in filename:
                model_type = 'lightgbm'
            
            # Determine target
            target = parts[1] if len(parts) > 1 else 'unknown'
            
            # Determine date
            date_part = parts[-1]
            
            # Create a unique key for this model
            model_key = f"{model_type}_{target}_{date_part}"
            
            # Store metrics
            all_metrics[model_key] = {
                'model_type': model_type,
                'target': target,
                'date': date_part,
                'metrics': metrics
            }
            
        except Exception as e:
            logging.error(f"Error loading metrics from {metrics_file}: {str(e)}")
    
    return all_metrics

def compare_models(models_dir=None, output_dir=None):
    """
    Compare performance of different models
    
    Args:
        models_dir (str, optional): Path to models directory
        output_dir (str, optional): Path to output directory for visualizations
        
    Returns:
        pd.DataFrame: DataFrame with model comparison metrics
    """
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                 'data', 'visualizations')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load metrics for all models
    all_metrics = load_model_metrics(models_dir)
    
    # Create a DataFrame to compare models
    comparison_data = []
    
    for model_key, model_info in all_metrics.items():
        model_type = model_info['model_type']
        target = model_info['target']
        date = model_info['date']
        metrics = model_info['metrics']
        
        # Extract key metrics
        r2 = metrics.get('test_r2', 0)
        rmse = metrics.get('test_rmse', 0)
        mae = metrics.get('test_mae', 0)
        
        comparison_data.append({
            'model_key': model_key,
            'model_type': model_type,
            'target': target,
            'date': date,
            'R2': r2,
            'RMSE': rmse,
            'MAE': mae
        })
    
    # Convert to DataFrame
    df = pd.DataFrame(comparison_data)
    
    # If the DataFrame is empty, return it
    if df.empty:
        logging.warning("No model metrics found")
        return df
    
    # Group by target and model_type, taking the latest model of each type
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d', errors='coerce')
    latest_models = df.sort_values('date', ascending=False).groupby(['target', 'model_type']).first().reset_index()
    
    # Generate visualizations
    _generate_model_comparison_plots(latest_models, output_dir)
    
    # Return the full comparison DataFrame
    return df

def _generate_model_comparison_plots(df, output_dir):
    """
    Generate plots comparing different model types
    
    Args:
        df (pd.DataFrame): DataFrame with model comparison metrics
        output_dir (str): Path to output directory for visualizations
    """
    # Set up the style
    sns.set(style='whitegrid')
    
    # 1. R² comparison by target and model type
    plt.figure(figsize=(12, 8))
    chart = sns.barplot(x='target', y='R2', hue='model_type', data=df)
    chart.set_xlabel('Target Statistic')
    chart.set_ylabel('R² Score')
    chart.set_title('Model R² Score Comparison by Target and Model Type')
    chart.set_xticklabels(chart.get_xticklabels(), rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_r2_comparison.png'))
    plt.close()
    
    # 2. RMSE comparison by target and model type
    plt.figure(figsize=(12, 8))
    chart = sns.barplot(x='target', y='RMSE', hue='model_type', data=df)
    chart.set_xlabel('Target Statistic')
    chart.set_ylabel('RMSE')
    chart.set_title('Model RMSE Comparison by Target and Model Type')
    chart.set_xticklabels(chart.get_xticklabels(), rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_rmse_comparison.png'))
    plt.close()
    
    # 3. MAE comparison by target and model type
    plt.figure(figsize=(12, 8))
    chart = sns.barplot(x='target', y='MAE', hue='model_type', data=df)
    chart.set_xlabel('Target Statistic')
    chart.set_ylabel('MAE')
    chart.set_title('Model MAE Comparison by Target and Model Type')
    chart.set_xticklabels(chart.get_xticklabels(), rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_mae_comparison.png'))
    plt.close()
    
    # 4. Relative improvement (if multiple model types for the same target)
    if len(df['model_type'].unique()) > 1:
        improvement_data = []
        
        for target in df['target'].unique():
            target_df = df[df['target'] == target]
            
            # Find the baseline model (random forest)
            baseline = target_df[target_df['model_type'] == 'random_forest']
            if len(baseline) == 0:
                # If no random forest, use the first model as baseline
                baseline = target_df.iloc[0:1]
            
            baseline_r2 = baseline['R2'].values[0]
            baseline_rmse = baseline['RMSE'].values[0]
            baseline_mae = baseline['MAE'].values[0]
            
            for _, row in target_df.iterrows():
                if row['model_type'] != 'random_forest':
                    # Calculate relative improvements
                    r2_improvement = (row['R2'] - baseline_r2) / max(0.001, baseline_r2) * 100
                    rmse_improvement = (baseline_rmse - row['RMSE']) / max(0.001, baseline_rmse) * 100
                    mae_improvement = (baseline_mae - row['MAE']) / max(0.001, baseline_mae) * 100
                    
                    improvement_data.append({
                        'target': target,
                        'model_type': row['model_type'],
                        'baseline': 'random_forest',
                        'R2_improvement': r2_improvement,
                        'RMSE_improvement': rmse_improvement,
                        'MAE_improvement': mae_improvement
                    })
        
        if improvement_data:
            improvement_df = pd.DataFrame(improvement_data)
            
            # Plot R² improvement
            plt.figure(figsize=(12, 8))
            chart = sns.barplot(x='target', y='R2_improvement', hue='model_type', data=improvement_df)
            chart.set_xlabel('Target Statistic')
            chart.set_ylabel('R² Improvement (%)')
            chart.set_title('R² Improvement Over Random Forest by Target')
            chart.set_xticklabels(chart.get_xticklabels(), rotation=45)
            chart.yaxis.set_major_formatter(PercentFormatter())
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'r2_improvement.png'))
            plt.close()
            
            # Plot RMSE improvement
            plt.figure(figsize=(12, 8))
            chart = sns.barplot(x='target', y='RMSE_improvement', hue='model_type', data=improvement_df)
            chart.set_xlabel('Target Statistic')
            chart.set_ylabel('RMSE Reduction (%)')
            chart.set_title('RMSE Reduction Over Random Forest by Target')
            chart.set_xticklabels(chart.get_xticklabels(), rotation=45)
            chart.yaxis.set_major_formatter(PercentFormatter())
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'rmse_improvement.png'))
            plt.close()
            
            # Plot MAE improvement
            plt.figure(figsize=(12, 8))
            chart = sns.barplot(x='target', y='MAE_improvement', hue='model_type', data=improvement_df)
            chart.set_xlabel('Target Statistic')
            chart.set_ylabel('MAE Reduction (%)')
            chart.set_title('MAE Reduction Over Random Forest by Target')
            chart.set_xticklabels(chart.get_xticklabels(), rotation=45)
            chart.yaxis.set_major_formatter(PercentFormatter())
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'mae_improvement.png'))
            plt.close()

def visualize_feature_importance(models_dir=None, output_dir=None, model_type='all'):
    """
    Visualize feature importance across different models
    
    Args:
        models_dir (str, optional): Path to models directory
        output_dir (str, optional): Path to output directory for visualizations
        model_type (str): Which model type to visualize ('all', 'random_forest', 'xgboost', etc.)
        
    Returns:
        dict: Dictionary of feature importance DataFrames by target
    """
    if models_dir is None:
        models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
    
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                 'data', 'visualizations')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all feature importance files
    importance_files = []
    for file in os.listdir(models_dir):
        if file.endswith('.json') and 'feature_importance' in file:
            # Filter by model type if specified
            if model_type != 'all':
                if model_type.lower() not in file.lower():
                    continue
            importance_files.append(os.path.join(models_dir, file))
    
    # Sort by filename (which includes date)
    importance_files.sort()
    
    # Process each importance file
    all_importances = {}
    
    for imp_file in importance_files:
        try:
            with open(imp_file, 'r') as f:
                importance = json.load(f)
            
            # Extract model details from filename
            filename = os.path.basename(imp_file)
            parts = filename.replace('.json', '').split('_')
            
            # Determine target
            target_idx = parts.index('importance') + 1
            target = parts[target_idx] if target_idx < len(parts) else 'unknown'
            
            # Convert to DataFrame
            if isinstance(importance, dict):
                # Handle dictionary format
                df = pd.DataFrame(list(importance.items()), columns=['feature', 'importance'])
            elif isinstance(importance, list):
                # Handle list of (feature, importance) tuples
                df = pd.DataFrame(importance, columns=['feature', 'importance'])
            else:
                logging.warning(f"Unrecognized importance format in {imp_file}")
                continue
            
            # Sort by importance
            df = df.sort_values('importance', ascending=False)
            
            # Store
            all_importances[target] = df
            
        except Exception as e:
            logging.error(f"Error loading feature importance from {imp_file}: {str(e)}")
    
    # Generate visualizations for each target
    for target, imp_df in all_importances.items():
        # Take top 20 features
        plot_df = imp_df.head(20)
        
        plt.figure(figsize=(12, 10))
        chart = sns.barplot(x='importance', y='feature', data=plot_df)
        chart.set_xlabel('Feature Importance')
        chart.set_ylabel('Feature')
        chart.set_title(f'Top 20 Features for {target.upper()} Prediction')
        plt.tight_layout()
        
        # Save with model type in filename if specified
        if model_type != 'all':
            plt.savefig(os.path.join(output_dir, f'feature_importance_{target}_{model_type}.png'))
        else:
            plt.savefig(os.path.join(output_dir, f'feature_importance_{target}.png'))
        plt.close()
    
    return all_importances

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare model performance and visualize results")
    
    parser.add_argument("--models-dir", type=str,
                        help="Path to models directory")
    parser.add_argument("--output-dir", type=str,
                        help="Path to output directory for visualizations")
    parser.add_argument("--feature-importance", action="store_true",
                        help="Generate feature importance visualizations")
    parser.add_argument("--model-type", type=str, default="all",
                        choices=["all", "random_forest", "xgboost", "gradient_boosting", "lightgbm"],
                        help="Model type to visualize feature importance for")
    
    args = parser.parse_args()
    
    # Compare models
    df = compare_models(args.models_dir, args.output_dir)
    print("Model comparison completed. Dataframe shape:", df.shape)
    
    # Generate feature importance visualizations if requested
    if args.feature_importance:
        importances = visualize_feature_importance(args.models_dir, args.output_dir, args.model_type)
        print(f"Feature importance visualizations generated for {len(importances)} targets")