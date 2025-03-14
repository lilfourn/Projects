#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module for visualizing NBA prediction model results
"""

import os
import sys
import logging
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from plotly.subplots import make_subplots

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Constants
CURRENT_DATE = datetime.now().strftime('%Y%m%d')
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models")
VISUALIZATION_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "visualizations")
os.makedirs(VISUALIZATION_DIR, exist_ok=True)

# Create a function to get the visualization directory for a specific target
def get_target_viz_dir(target_name):
    """
    Get the visualization directory for a specific target
    
    Args:
        target_name (str): Target variable name (e.g., pts, reb, ast)
        
    Returns:
        str: Path to the target-specific visualization directory
    """
    if target_name is None:
        return VISUALIZATION_DIR
    
    # Map common target names to stat categories
    stat_categories = {
        'pts': 'scoring',
        'reb': 'rebounds',
        'ast': 'assists',
        'stl': 'defense',
        'blk': 'defense',
        'tov': 'miscellaneous',
        'pf': 'miscellaneous',
        'fg3m': 'scoring',
        'fgm': 'scoring',
        'ftm': 'scoring',
        'fg3a': 'scoring',
        'fga': 'scoring',
        'fta': 'scoring',
        'oreb': 'rebounds',
        'dreb': 'rebounds',
        'min': 'minutes',
    }
    
    # Get the category for the target, default to 'other' if not found
    category = stat_categories.get(target_name.lower(), 'other')
    
    # Create the category directory
    category_dir = os.path.join(VISUALIZATION_DIR, category)
    os.makedirs(category_dir, exist_ok=True)
    
    # Create the target directory within the category
    target_dir = os.path.join(category_dir, target_name.lower())
    os.makedirs(target_dir, exist_ok=True)
    
    return target_dir

# Set the plotting style for consistent visualizations
def set_plotting_style():
    """Set the plotting style for consistent visualizations"""
    sns.set(style="whitegrid")
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['figure.titlesize'] = 20

def visualize_feature_importance(target_name, date_str=None, top_n=20, save_fig=True, show_fig=True):
    """
    Create a visualization of feature importance for a specific target model
    
    Args:
        target_name (str): Name of the target variable (e.g., 'pts', 'reb', 'ast')
        date_str (str, optional): Date string for the model file. If None, will use the latest.
        top_n (int): Number of top features to display
        save_fig (bool): Whether to save the figure to disk
        show_fig (bool): Whether to display the figure
        
    Returns:
        str: Path to the saved figure, or None if not saved
    """
    # Set the plotting style
    set_plotting_style()
    
    # Find the feature importance file
    target_dir = os.path.join(MODELS_DIR, target_name)
    if not os.path.exists(target_dir):
        logging.error(f"Target directory not found: {target_dir}")
        return None
        
    if date_str is None:
        # Find the latest feature importance file for this target
        import glob
        feature_files = glob.glob(os.path.join(target_dir, f"*_feature_importance_*.json"))
        if not feature_files:
            logging.error(f"No feature importance files found for target: {target_name}")
            return None
        
        # Sort by modification time (newest first)
        feature_files.sort(key=os.path.getmtime, reverse=True)
        feature_file = feature_files[0]
        # Extract date from filename
        date_str = os.path.basename(feature_file).split('_')[-1].split('.')[0]
    else:
        # Look for any feature importance file with the given date
        import glob
        feature_files = glob.glob(os.path.join(target_dir, f"*_feature_importance_{date_str}.json"))
        if not feature_files:
            logging.error(f"Feature importance file not found for date: {date_str}")
            return None
        feature_file = feature_files[0]
    
    # Load the feature importance data
    try:
        with open(feature_file, 'r') as f:
            importance_data = json.load(f)
    except Exception as e:
        logging.error(f"Error loading feature importance data: {str(e)}")
        return None
    
    # Convert to DataFrame for easier plotting
    df = pd.DataFrame(list(importance_data.items()), columns=['Feature', 'Importance'])
    df = df.sort_values('Importance', ascending=False).head(top_n)
    
    # Create a horizontal bar chart
    plt.figure(figsize=(12, 10))
    ax = sns.barplot(x='Importance', y='Feature', data=df, palette='viridis')
    
    # Add value labels to the bars
    for i, v in enumerate(df['Importance']):
        ax.text(v + 0.01, i, f"{v:.4f}", va='center')
    
    # Set the title and labels
    plt.title(f"Top {top_n} Features for {target_name.upper()} Prediction Model", fontweight='bold')
    plt.xlabel('Relative Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    
    # Save the figure if requested
    if save_fig:
        # Create a professional filename
        filename = f"feature_importance_{target_name}_top{top_n}_{date_str}.png"
        save_path = os.path.join(get_target_viz_dir(target_name), filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"Saved feature importance visualization to {save_path}")
    
    # Show the figure if requested
    if show_fig:
        plt.show()
    else:
        plt.close()
    
    # Create an interactive Plotly version
    if save_fig:
        # Create a horizontal bar chart with Plotly
        fig = px.bar(
            df, 
            x='Importance', 
            y='Feature', 
            orientation='h',
            title=f"Top {top_n} Features for {target_name.upper()} Prediction Model",
            labels={'Importance': 'Relative Importance', 'Feature': 'Feature'},
            color='Importance',
            color_continuous_scale='Viridis'
        )
        
        # Update layout for a professional look
        fig.update_layout(
            font=dict(family="Arial, sans-serif", size=14),
            title_font=dict(size=20, family="Arial, sans-serif"),
            title_x=0.5,  # Center the title
            xaxis_title_font=dict(size=16),
            yaxis_title_font=dict(size=16),
            margin=dict(l=150),  # Add margin for feature names
            coloraxis_colorbar=dict(title="Importance")
        )
        
        # Save as interactive HTML
        html_filename = f"feature_importance_{target_name}_top{top_n}_{date_str}.html"
        html_save_path = os.path.join(get_target_viz_dir(target_name), html_filename)
        fig.write_html(html_save_path)
        logging.info(f"Saved interactive feature importance visualization to {html_save_path}")
        
        return save_path
    
    return None

def visualize_model_metrics(target_name=None, date_str=None, save_fig=True, show_fig=True):
    """
    Create a visualization of model metrics for one or all target models
    
    Args:
        target_name (str, optional): Name of the target variable. If None, will visualize all targets.
        date_str (str, optional): Date string for the model file. If None, will use the latest.
        save_fig (bool): Whether to save the figure to disk
        show_fig (bool): Whether to display the figure
        
    Returns:
        str: Path to the saved figure, or None if not saved
    """
    # Set the plotting style
    set_plotting_style()
    
    # Find the metrics files
    if target_name is None:
        # Find all metrics files
        import glob
        metrics_files = glob.glob(os.path.join(MODELS_DIR, "nba_*_metrics_*.json"))
        if not metrics_files:
            logging.error("No metrics files found")
            return None
        
        # Extract target names from filenames
        targets = list(set([os.path.basename(f).split('_')[1] for f in metrics_files]))
        logging.info(f"Found metrics for targets: {targets}")
        
        # If date_str is provided, filter to only that date
        if date_str is not None:
            metrics_files = [f for f in metrics_files if date_str in f]
            if not metrics_files:
                logging.error(f"No metrics files found for date: {date_str}")
                return None
    else:
        # Find metrics files for the specific target
        if date_str is None:
            metrics_files = glob.glob(os.path.join(MODELS_DIR, f"nba_{target_name}_metrics_*.json"))
            if not metrics_files:
                logging.error(f"No metrics files found for target: {target_name}")
                return None
            
            # Sort by modification time (newest first)
            metrics_files.sort(key=os.path.getmtime, reverse=True)
            # Extract date from filename
            date_str = os.path.basename(metrics_files[0]).split('_')[-1].split('.')[0]
        else:
            metrics_file = os.path.join(MODELS_DIR, f"nba_{target_name}_metrics_{date_str}.json")
            if not os.path.exists(metrics_file):
                logging.error(f"Metrics file not found: {metrics_file}")
                return None
            metrics_files = [metrics_file]
    
    # Load the metrics data
    metrics_data = {}
    for file in metrics_files:
        try:
            with open(file, 'r') as f:
                data = json.load(f)
            
            # Extract target name from filename
            target = os.path.basename(file).split('_')[1]
            metrics_data[target] = data
        except Exception as e:
            logging.error(f"Error loading metrics data from {file}: {str(e)}")
    
    if not metrics_data:
        logging.error("No metrics data could be loaded")
        return None
    
    # Create a DataFrame for plotting
    metrics_list = []
    for target, data in metrics_data.items():
        # Extract the key metrics
        metrics = {
            'Target': target.upper(),
            'R²': data.get('r2', 0),
            'MAE': data.get('mae', 0),
            'RMSE': data.get('rmse', 0),
            'CV R²': data.get('cv_r2_mean', 0)
        }
        metrics_list.append(metrics)
    
    df = pd.DataFrame(metrics_list)
    
    # Create a grouped bar chart
    plt.figure(figsize=(14, 10))
    
    # Set up the plot
    x = np.arange(len(df['Target']))
    width = 0.2
    
    # Plot each metric
    plt.bar(x - width*1.5, df['R²'], width, label='R²', color='#1f77b4')
    plt.bar(x - width/2, df['CV R²'], width, label='CV R²', color='#ff7f0e')
    plt.bar(x + width/2, df['MAE'], width, label='MAE', color='#2ca02c')
    plt.bar(x + width*1.5, df['RMSE'], width, label='RMSE', color='#d62728')
    
    # Add labels and title
    plt.xlabel('Target Variable')
    plt.ylabel('Metric Value')
    plt.title('Model Performance Metrics by Target Variable', fontweight='bold')
    plt.xticks(x, df['Target'])
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on top of bars
    for i, metric in enumerate(['R²', 'CV R²', 'MAE', 'RMSE']):
        positions = x - width*1.5 + width*i
        for j, pos in enumerate(positions):
            value = df[metric].iloc[j]
            plt.text(pos, value + 0.02, f"{value:.3f}", ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    # Save the figure if requested
    if save_fig:
        # Create a professional filename
        if target_name is None:
            filename = f"model_metrics_comparison_{date_str}.png"
        else:
            filename = f"model_metrics_{target_name}_{date_str}.png"
        
        save_path = os.path.join(get_target_viz_dir(target_name), filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"Saved model metrics visualization to {save_path}")
    
    # Show the figure if requested
    if show_fig:
        plt.show()
    else:
        plt.close()
    
    # Create an interactive Plotly version
    if save_fig:
        # Create a grouped bar chart with Plotly
        fig = go.Figure()
        
        # Add each metric as a separate trace
        fig.add_trace(go.Bar(
            x=df['Target'],
            y=df['R²'],
            name='R²',
            marker_color='#1f77b4'
        ))
        
        fig.add_trace(go.Bar(
            x=df['Target'],
            y=df['CV R²'],
            name='CV R²',
            marker_color='#ff7f0e'
        ))
        
        fig.add_trace(go.Bar(
            x=df['Target'],
            y=df['MAE'],
            name='MAE',
            marker_color='#2ca02c'
        ))
        
        fig.add_trace(go.Bar(
            x=df['Target'],
            y=df['RMSE'],
            name='RMSE',
            marker_color='#d62728'
        ))
        
        # Update layout for a professional look
        fig.update_layout(
            title='Model Performance Metrics by Target Variable',
            xaxis_title='Target Variable',
            yaxis_title='Metric Value',
            barmode='group',
            font=dict(family="Arial, sans-serif", size=14),
            title_font=dict(size=20, family="Arial, sans-serif"),
            title_x=0.5,  # Center the title
            legend=dict(
                x=0.01,
                y=0.99,
                bgcolor='rgba(255, 255, 255, 0.5)',
                bordercolor='rgba(0, 0, 0, 0.5)'
            ),
            template='plotly_white'
        )
        
        # Add annotations for the values
        for i, target in enumerate(df['Target']):
            metrics = ['R²', 'CV R²', 'MAE', 'RMSE']
            positions = [-0.3, -0.1, 0.1, 0.3]  # Adjust these based on the bar positions
            
            for j, metric in enumerate(metrics):
                value = df[metric].iloc[i]
                fig.add_annotation(
                    x=target,
                    y=value,
                    text=f"{value:.3f}",
                    showarrow=False,
                    xshift=positions[j] * 40,
                    yshift=10,
                    font=dict(size=10)
                )
        
        # Save as interactive HTML
        if target_name is None:
            html_filename = f"model_metrics_comparison_{date_str}.html"
        else:
            html_filename = f"model_metrics_{target_name}_{date_str}.html"
            
        html_save_path = os.path.join(get_target_viz_dir(target_name), html_filename)
        fig.write_html(html_save_path)
        logging.info(f"Saved interactive model metrics visualization to {html_save_path}")
        
        return save_path
    
    return None

def visualize_prediction_analysis(target_name, actual, predicted, date_str=None, save_fig=True, show_fig=True):
    """
    Create visualizations to analyze model predictions vs actual values
    
    Args:
        target_name (str): Name of the target variable
        actual (array-like): Actual values
        predicted (array-like): Predicted values
        date_str (str, optional): Date string for the filename. If None, will use current date.
        save_fig (bool): Whether to save the figure to disk
        show_fig (bool): Whether to display the figure
        
    Returns:
        str: Path to the saved figure, or None if not saved
    """
    # Set the plotting style
    set_plotting_style()
    
    # Use current date if not provided
    if date_str is None:
        date_str = datetime.now().strftime("%Y%m%d")
    
    # Create a figure with multiple subplots
    fig, axs = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle(f"Prediction Analysis for {target_name.upper()}", fontsize=24, fontweight='bold')
    
    # 1. Scatter plot of actual vs predicted values
    axs[0, 0].scatter(actual, predicted, alpha=0.5)
    axs[0, 0].plot([min(actual), max(actual)], [min(actual), max(actual)], 'r--')
    axs[0, 0].set_xlabel('Actual Values')
    axs[0, 0].set_ylabel('Predicted Values')
    axs[0, 0].set_title('Actual vs Predicted Values')
    axs[0, 0].grid(True, linestyle='--', alpha=0.7)
    
    # Add correlation coefficient
    corr = np.corrcoef(actual, predicted)[0, 1]
    axs[0, 0].annotate(f"Correlation: {corr:.4f}", 
                      xy=(0.05, 0.95), xycoords='axes fraction',
                      bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    # 2. Histogram of prediction errors
    errors = predicted - actual
    axs[0, 1].hist(errors, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axs[0, 1].axvline(x=0, color='r', linestyle='--')
    axs[0, 1].set_xlabel('Prediction Error')
    axs[0, 1].set_ylabel('Frequency')
    axs[0, 1].set_title('Distribution of Prediction Errors')
    axs[0, 1].grid(True, linestyle='--', alpha=0.7)
    
    # Add mean and std of errors
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    axs[0, 1].annotate(f"Mean: {mean_error:.4f}\nStd Dev: {std_error:.4f}", 
                      xy=(0.05, 0.95), xycoords='axes fraction',
                      bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    # 3. Residual plot
    axs[1, 0].scatter(predicted, errors, alpha=0.5)
    axs[1, 0].axhline(y=0, color='r', linestyle='--')
    axs[1, 0].set_xlabel('Predicted Values')
    axs[1, 0].set_ylabel('Residuals')
    axs[1, 0].set_title('Residual Plot')
    axs[1, 0].grid(True, linestyle='--', alpha=0.7)
    
    # 4. Q-Q plot for residuals
    from scipy import stats
    stats.probplot(errors, dist="norm", plot=axs[1, 1])
    axs[1, 1].set_title('Q-Q Plot of Residuals')
    axs[1, 1].grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for the suptitle
    
    # Save the figure if requested
    if save_fig:
        # Create a professional filename
        filename = f"prediction_analysis_{target_name}_{date_str}.png"
        save_path = os.path.join(get_target_viz_dir(target_name), filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"Saved prediction analysis visualization to {save_path}")
    
    # Show the figure if requested
    if show_fig:
        plt.show()
    else:
        plt.close()
    
    # Create an interactive Plotly version
    if save_fig:
        # Create a figure with subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Actual vs Predicted Values',
                'Distribution of Prediction Errors',
                'Residual Plot',
                'Cumulative Distribution of Errors'
            ),
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )
        
        # 1. Scatter plot of actual vs predicted
        fig.add_trace(
            go.Scatter(
                x=actual, y=predicted,
                mode='markers',
                marker=dict(color='blue', opacity=0.5),
                name='Data Points'
            ),
            row=1, col=1
        )
        
        # Add identity line
        min_val = min(min(actual), min(predicted))
        max_val = max(max(actual), max(predicted))
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val], y=[min_val, max_val],
                mode='lines',
                line=dict(color='red', dash='dash'),
                name='Identity Line'
            ),
            row=1, col=1
        )
        
        # 2. Histogram of errors
        fig.add_trace(
            go.Histogram(
                x=errors,
                marker=dict(color='skyblue', line=dict(color='black', width=1)),
                name='Error Distribution'
            ),
            row=1, col=2
        )
        
        # Add vertical line at zero
        fig.add_trace(
            go.Scatter(
                x=[0, 0], y=[0, 30],  # Y values will be scaled automatically
                mode='lines',
                line=dict(color='red', dash='dash'),
                name='Zero Error',
                showlegend=False
            ),
            row=1, col=2
        )
        
        # 3. Residual plot
        fig.add_trace(
            go.Scatter(
                x=predicted, y=errors,
                mode='markers',
                marker=dict(color='green', opacity=0.5),
                name='Residuals'
            ),
            row=2, col=1
        )
        
        # Add horizontal line at zero
        fig.add_trace(
            go.Scatter(
                x=[min(predicted), max(predicted)], y=[0, 0],
                mode='lines',
                line=dict(color='red', dash='dash'),
                name='Zero Line',
                showlegend=False
            ),
            row=2, col=1
        )
        
        # 4. Cumulative distribution of errors
        sorted_errors = np.sort(errors)
        p = np.linspace(0, 1, len(sorted_errors))
        
        fig.add_trace(
            go.Scatter(
                x=sorted_errors, y=p,
                mode='lines',
                line=dict(color='purple'),
                name='Empirical CDF'
            ),
            row=2, col=2
        )
        
        # Add annotations for statistics
        fig.add_annotation(
            text=f"Correlation: {corr:.4f}",
            xref="x1", yref="y1",
            x=0.05, y=0.95,
            xanchor="left", yanchor="top",
            showarrow=False,
            bgcolor="white",
            bordercolor="gray",
            borderwidth=1,
            borderpad=4,
            font=dict(size=12)
        )
        
        fig.add_annotation(
            text=f"Mean Error: {mean_error:.4f}<br>Std Dev: {std_error:.4f}",
            xref="x2", yref="y2",
            x=0.05, y=0.95,
            xanchor="left", yanchor="top",
            showarrow=False,
            bgcolor="white",
            bordercolor="gray",
            borderwidth=1,
            borderpad=4,
            font=dict(size=12)
        )
        
        # Update layout for a professional look
        fig.update_layout(
            title=f"Prediction Analysis for {target_name.upper()}",
            font=dict(family="Arial, sans-serif", size=14),
            title_font=dict(size=24, family="Arial, sans-serif", color="black"),
            title_x=0.5,  # Center the title
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            template='plotly_white'
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Actual Values", row=1, col=1)
        fig.update_yaxes(title_text="Predicted Values", row=1, col=1)
        
        fig.update_xaxes(title_text="Prediction Error", row=1, col=2)
        fig.update_yaxes(title_text="Frequency", row=1, col=2)
        
        fig.update_xaxes(title_text="Predicted Values", row=2, col=1)
        fig.update_yaxes(title_text="Residuals", row=2, col=1)
        
        fig.update_xaxes(title_text="Error Value", row=2, col=2)
        fig.update_yaxes(title_text="Cumulative Probability", row=2, col=2)
        
        # Save as interactive HTML
        html_filename = f"prediction_analysis_{target_name}_{date_str}.html"
        html_save_path = os.path.join(get_target_viz_dir(target_name), html_filename)
        fig.write_html(html_save_path)
        logging.info(f"Saved interactive prediction analysis visualization to {html_save_path}")
        
        return save_path
    
    return None

def visualize_model_comparison(comparison_data, model_type=None, date_str=None, save_fig=True, show_fig=True):
    """
    Create a visualization comparing model performance across different targets
    
    Args:
        comparison_data (dict): Dictionary with target names as keys and metrics as values
        model_type (str, optional): Type of model used (for title)
        date_str (str, optional): Date string for the filename. If None, will use current date.
        save_fig (bool): Whether to save the figure to disk
        show_fig (bool): Whether to display the figure
        
    Returns:
        str: Path to the saved figure, or None if not saved
    """
    # Set the plotting style
    set_plotting_style()
    
    if not comparison_data:
        logging.error("No comparison data provided")
        return None
    
    # Use current date if not provided
    if date_str is None:
        date_str = CURRENT_DATE
    
    # Create a DataFrame from the comparison data
    metrics_list = []
    for target, metrics in comparison_data.items():
        metrics_dict = {
            'Target': target.upper(),
            'R²': metrics.get('r2', 0),
            'MAE': metrics.get('mae', 0),
            'RMSE': metrics.get('rmse', 0)
        }
        metrics_list.append(metrics_dict)
    
    df = pd.DataFrame(metrics_list)
    
    # Create a figure with 3 subplots (one for each metric)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot R² scores
    sns.barplot(x='Target', y='R²', data=df, ax=axes[0], palette='viridis')
    axes[0].set_title('R² Score by Target', fontweight='bold')
    axes[0].set_ylim(0, 1)  # R² is typically between 0 and 1
    
    # Add value labels
    for i, v in enumerate(df['R²']):
        axes[0].text(i, v + 0.02, f"{v:.4f}", ha='center')
    
    # Plot MAE
    sns.barplot(x='Target', y='MAE', data=df, ax=axes[1], palette='viridis')
    axes[1].set_title('Mean Absolute Error by Target', fontweight='bold')
    
    # Add value labels
    for i, v in enumerate(df['MAE']):
        axes[1].text(i, v + 0.02, f"{v:.4f}", ha='center')
    
    # Plot RMSE
    sns.barplot(x='Target', y='RMSE', data=df, ax=axes[2], palette='viridis')
    axes[2].set_title('Root Mean Squared Error by Target', fontweight='bold')
    
    # Add value labels
    for i, v in enumerate(df['RMSE']):
        axes[2].text(i, v + 0.02, f"{v:.4f}", ha='center')
    
    # Set the main title
    if model_type:
        fig.suptitle(f"Model Performance Comparison ({model_type.upper()})", fontsize=20, fontweight='bold')
    else:
        fig.suptitle("Model Performance Comparison", fontsize=20, fontweight='bold')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    
    # Determine the appropriate directory for saving
    # For comparisons across multiple targets, we'll save in a 'comparisons' directory
    comparison_dir = os.path.join(VISUALIZATION_DIR, 'comparisons')
    os.makedirs(comparison_dir, exist_ok=True)
    
    # If comparing different model types for a single target, save in that target's directory
    if len(comparison_data) == 1:
        target_name = list(comparison_data.keys())[0]
        save_dir = get_target_viz_dir(target_name)
    else:
        save_dir = comparison_dir
    
    # Save the figure if requested
    if save_fig:
        # Create a professional filename
        model_type_str = f"{model_type}_" if model_type else ""
        filename = f"model_comparison_{model_type_str}{date_str}.png"
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"Saved model comparison visualization to {save_path}")
    
    # Show the figure if requested
    if show_fig:
        plt.show()
    else:
        plt.close()
    
    # Create an interactive Plotly version
    if save_fig:
        # Melt the DataFrame for easier plotting with Plotly
        df_melted = pd.melt(df, id_vars=['Target'], value_vars=['R²', 'MAE', 'RMSE'], 
                           var_name='Metric', value_name='Value')
        
        # Create a grouped bar chart
        fig = px.bar(
            df_melted, 
            x='Target', 
            y='Value', 
            color='Metric',
            barmode='group',
            title=f"Model Performance Comparison {f'({model_type.upper()})' if model_type else ''}",
            labels={'Value': 'Metric Value', 'Target': 'Target Variable'},
            color_discrete_sequence=px.colors.qualitative.Plotly
        )
        
        # Update layout for a professional look
        fig.update_layout(
            font=dict(family="Arial, sans-serif", size=14),
            title_font=dict(size=20, family="Arial, sans-serif"),
            title_x=0.5,  # Center the title
            xaxis_title_font=dict(size=16),
            yaxis_title_font=dict(size=16),
            legend_title_font=dict(size=14),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Add value annotations
        for i, row in df_melted.iterrows():
            fig.add_annotation(
                x=row['Target'],
                y=row['Value'],
                text=f"{row['Value']:.4f}",
                showarrow=False,
                yshift=10,
                font=dict(size=10)
            )
        
        # Save as interactive HTML
        model_type_str = f"{model_type}_" if model_type else ""
        html_filename = f"model_comparison_{model_type_str}{date_str}.html"
        html_save_path = os.path.join(save_dir, html_filename)
        fig.write_html(html_save_path)
        logging.info(f"Saved interactive model comparison visualization to {html_save_path}")
        
        return save_path
    
    return None

def main():
    """Main function to generate visualizations"""
    import argparse
    import glob
    
    parser = argparse.ArgumentParser(description='Generate model visualizations')
    parser.add_argument('--target', type=str, help='Target variable to visualize (e.g., pts, reb, ast)')
    parser.add_argument('--date', type=str, help='Date string for model files (YYYYMMDD format)')
    parser.add_argument('--top-n', type=int, default=20, help='Number of top features to display')
    parser.add_argument('--show', action='store_true', help='Show visualizations (default: False)')
    parser.add_argument('--no-save', action='store_true', help='Do not save visualizations (default: False)')
    parser.add_argument('--type', type=str, choices=['feature_importance', 'metrics', 'predictions', 'comparison', 'all'],
                       default='all', help='Type of visualization to generate')
    parser.add_argument('--model-type', type=str, help='Model type (e.g., random_forest, xgboost, lightgbm)')
    parser.add_argument('--compare-targets', type=str, nargs='+', help='List of targets to compare')
    
    args = parser.parse_args()
    
    # Set save and show flags
    save_fig = not args.no_save
    show_fig = args.show
    
    # Use current date if not provided
    date_str = args.date if args.date else CURRENT_DATE
    
    # Generate visualizations based on type
    if args.type == 'feature_importance' or args.type == 'all':
        if args.target:
            logging.info(f"Generating feature importance visualization for {args.target}")
            visualize_feature_importance(
                target_name=args.target,
                date_str=date_str,
                top_n=args.top_n,
                save_fig=save_fig,
                show_fig=show_fig
            )
        else:
            # Find all target models
            feature_files = glob.glob(os.path.join(MODELS_DIR, "feature_importance_*_*.json"))
            targets = list(set([os.path.basename(f).split('_')[1] for f in feature_files]))
            
            for target in targets:
                logging.info(f"Generating feature importance visualization for {target}")
                visualize_feature_importance(
                    target_name=target,
                    date_str=date_str,
                    top_n=args.top_n,
                    save_fig=save_fig,
                    show_fig=show_fig
                )
    
    if args.type == 'metrics' or args.type == 'all':
        if args.target:
            logging.info(f"Generating metrics visualization for {args.target}")
            visualize_model_metrics(
                target_name=args.target,
                date_str=date_str,
                save_fig=save_fig,
                show_fig=show_fig
            )
        else:
            logging.info("Generating metrics visualization for all targets")
            visualize_model_metrics(
                target_name=None,
                date_str=date_str,
                save_fig=save_fig,
                show_fig=show_fig
            )
    
    if args.type == 'predictions' or args.type == 'all':
        if not args.target:
            logging.error("Target must be specified for prediction analysis visualization")
        else:
            # For prediction analysis, we need actual and predicted values
            # This would typically be done right after model training
            # Here we'll just log that this needs to be done separately
            logging.info(f"Prediction analysis visualization for {args.target} requires actual and predicted values")
            logging.info("Run this after model training or provide the values directly")
    
    if args.type == 'comparison' or args.type == 'all':
        if args.compare_targets:
            # If specific targets are provided for comparison
            targets_to_compare = args.compare_targets
            logging.info(f"Comparing models for targets: {', '.join(targets_to_compare)}")
            
            # Load metrics for each target
            comparison_data = {}
            for target in targets_to_compare:
                metrics_file = os.path.join(MODELS_DIR, f"nba_{target}_metrics_{date_str}.json")
                if os.path.exists(metrics_file):
                    try:
                        with open(metrics_file, 'r') as f:
                            metrics = json.load(f)
                        comparison_data[target] = metrics
                    except Exception as e:
                        logging.error(f"Error loading metrics for {target}: {str(e)}")
                else:
                    logging.warning(f"Metrics file not found for {target}: {metrics_file}")
            
            if comparison_data:
                visualize_model_comparison(
                    comparison_data=comparison_data,
                    model_type=args.model_type,
                    date_str=date_str,
                    save_fig=save_fig,
                    show_fig=show_fig
                )
            else:
                logging.error("No metrics data found for comparison")
        else:
            # If no specific targets are provided, compare all available targets
            metrics_files = glob.glob(os.path.join(MODELS_DIR, "nba_*_metrics_*.json"))
            if not metrics_files:
                logging.error("No metrics files found for comparison")
                return
            
            # Filter to only the specified date if provided
            if date_str:
                metrics_files = [f for f in metrics_files if date_str in f]
            
            # Extract target names from filenames
            targets = list(set([os.path.basename(f).split('_')[1] for f in metrics_files]))
            logging.info(f"Comparing models for all available targets: {', '.join(targets)}")
            
            # Load metrics for each target
            comparison_data = {}
            for target in targets:
                metrics_file = os.path.join(MODELS_DIR, f"nba_{target}_metrics_{date_str}.json")
                if os.path.exists(metrics_file):
                    try:
                        with open(metrics_file, 'r') as f:
                            metrics = json.load(f)
                        comparison_data[target] = metrics
                    except Exception as e:
                        logging.error(f"Error loading metrics for {target}: {str(e)}")
            
            if comparison_data:
                visualize_model_comparison(
                    comparison_data=comparison_data,
                    model_type=args.model_type,
                    date_str=date_str,
                    save_fig=save_fig,
                    show_fig=show_fig
                )
            else:
                logging.error("No metrics data found for comparison")

if __name__ == "__main__":
    main()
