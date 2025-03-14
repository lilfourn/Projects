#!/usr/bin/env python3
# Feature Importance Visualization Tool
# This script generates visualizations of feature importance data

import os
import json
import argparse
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import matplotlib.dates as mdates

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Set default paths
try:
    from config import DATA_DIR
except ImportError:
    DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

ENGINEERED_DATA_DIR = os.path.join(DATA_DIR, "engineered")
FEATURE_IMPORTANCE_FILE = os.path.join(ENGINEERED_DATA_DIR, "feature_importance.json")
VIZ_OUTPUT_DIR = os.path.join(DATA_DIR, "visualizations")
os.makedirs(VIZ_OUTPUT_DIR, exist_ok=True)

def load_feature_importance(file_path=None):
    """
    Load feature importance from file
    
    Args:
        file_path (str, optional): Path to feature importance JSON file
    
    Returns:
        dict: Feature importance dictionary
    """
    if file_path is None:
        file_path = FEATURE_IMPORTANCE_FILE
    
    if not os.path.exists(file_path):
        logging.error(f"Feature importance file not found: {file_path}")
        return {}
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            
            # Handle different JSON formats
            # Check if data has the new structure with target_importances and permutation_importance
            if "target_importances" in data and "permutation_importance" in data:
                # Extract permutation importance which is more reliable
                flat_importance = {}
                
                # Try to get permutation importance for all targets
                for target, importances in data["permutation_importance"].items():
                    for item in importances:
                        feature = item["feature"]
                        importance = item["importance_mean"]
                        # Use highest importance value if feature appears in multiple targets
                        if feature not in flat_importance or importance > flat_importance[feature]:
                            flat_importance[feature] = importance
                
                # If no permutation importance data, try target_importances
                if not flat_importance and "target_importances" in data:
                    for target, importances in data["target_importances"].items():
                        for item in importances:
                            feature = item["feature"]
                            importance = item["importance"]
                            # Use highest importance value if feature appears in multiple targets
                            if feature not in flat_importance or importance > flat_importance[feature]:
                                flat_importance[feature] = importance
                
                logging.info(f"Loaded feature importance with {len(flat_importance)} features")
                return flat_importance
            elif "feature_importance" in data:
                # Handle older format with feature_importance key
                data = data["feature_importance"]
                logging.info(f"Loaded feature importance with {len(data)} features")
                return data
            else:
                # Assume it's already a flat dictionary
                logging.info(f"Loaded feature importance with {len(data)} features")
                return data
    except Exception as e:
        logging.error(f"Error loading importance file {file_path}: {str(e)}")
        return {}

def load_historical_importance(dir_path=None):
    """
    Load historical feature importance data from all saved model metrics
    
    Args:
        dir_path (str, optional): Directory to search for feature importance files
    
    Returns:
        pd.DataFrame: DataFrame with feature importance history
    """
    if dir_path is None:
        models_dir = os.path.join(os.path.dirname(DATA_DIR), "models")
        if not os.path.exists(models_dir):
            logging.error(f"Models directory not found: {models_dir}")
            return pd.DataFrame()
    else:
        models_dir = dir_path
    
    # Find all feature importance JSON files
    importance_files = []
    for root, _, files in os.walk(models_dir):
        for file in files:
            if file.startswith('feature_importance_') and file.endswith('.json'):
                importance_files.append(os.path.join(root, file))
    
    if not importance_files:
        # Also try to find metrics files which may contain feature importance
        for root, _, files in os.walk(models_dir):
            for file in files:
                if file.startswith('nba_') and file.endswith('_metrics.json'):
                    importance_files.append(os.path.join(root, file))
    
    if not importance_files:
        logging.warning("No historical feature importance files found")
        return pd.DataFrame()
    
    # Load all files and combine data
    all_data = []
    
    for file_path in sorted(importance_files):
        try:
            # Parse date from filename
            date_str = None
            if 'feature_importance_' in file_path:
                date_str = os.path.basename(file_path).replace('feature_importance_', '').replace('.json', '')
            else:
                # Try to extract date from metrics filename
                parts = os.path.basename(file_path).split('_')
                for part in parts:
                    if part.isdigit() and len(part) == 8:  # YYYYMMDD format
                        date_str = part
                        break
            
            if date_str is None:
                # Use file modification time if date not found in filename
                mod_time = os.path.getmtime(file_path)
                date_str = datetime.fromtimestamp(mod_time).strftime("%Y%m%d")
            
            # Convert to datetime
            try:
                date = datetime.strptime(date_str, "%Y%m%d")
            except:
                # If parsing fails, use file modification time
                mod_time = os.path.getmtime(file_path)
                date = datetime.fromtimestamp(mod_time)
            
            # Load feature importance data
            with open(file_path, 'r') as f:
                data = json.load(f)
                
                # Handle different JSON formats
                # Check if data has the new structure with target_importances and permutation_importance
                if "target_importances" in data and "permutation_importance" in data:
                    # Extract permutation importance which is more reliable
                    flat_importance = {}
                    
                    # Try to get permutation importance for all targets
                    for target, importances in data["permutation_importance"].items():
                        for item in importances:
                            feature = item["feature"]
                            importance = item["importance_mean"]
                            # Use highest importance value if feature appears in multiple targets
                            if feature not in flat_importance or importance > flat_importance[feature]:
                                flat_importance[feature] = importance
                    
                    # If no permutation importance data, try target_importances
                    if not flat_importance and "target_importances" in data:
                        for target, importances in data["target_importances"].items():
                            for item in importances:
                                feature = item["feature"]
                                importance = item["importance"]
                                # Use highest importance value if feature appears in multiple targets
                                if feature not in flat_importance or importance > flat_importance[feature]:
                                    flat_importance[feature] = importance
                    
                    # Convert to DataFrame format
                    for feature, importance in flat_importance.items():
                        all_data.append({
                            "date": date,
                            "feature": feature,
                            "importance": float(importance)
                        })
                
                # Find feature importance in metrics file if needed
                elif "feature_importance" in data:
                    data = data["feature_importance"]
                    
                    # Convert to DataFrame format
                    for feature, importance in data.items():
                        all_data.append({
                            "date": date,
                            "feature": feature,
                            "importance": float(importance)
                        })
                        
                # Standard flat dictionary format
                else:
                    # Convert to DataFrame format
                    for feature, importance in data.items():
                        all_data.append({
                            "date": date,
                            "feature": feature,
                            "importance": float(importance)
                        })
                    
        except Exception as e:
            logging.warning(f"Error loading importance file {file_path}: {str(e)}")
    
    if not all_data:
        logging.warning("No valid feature importance data found")
        return pd.DataFrame()
    
    # Create DataFrame
    df = pd.DataFrame(all_data)
    logging.info(f"Loaded historical feature importance with {df['feature'].nunique()} features across {df['date'].nunique()} dates")
    return df

def plot_top_features(importance_data, top_n=20, output_file=None, show_plot=True):
    """
    Plot top N most important features
    
    Args:
        importance_data (dict or pd.DataFrame): Feature importance data
        top_n (int): Number of top features to display
        output_file (str, optional): Path to save the plot
        show_plot (bool): Whether to display the plot
    """
    plt.figure(figsize=(12, 10))
    
    if isinstance(importance_data, dict):
        # Convert dictionary to sorted list of (feature, importance) tuples
        sorted_features = sorted(importance_data.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        # Create Series for plotting
        feature_names = [f[0] for f in sorted_features]
        importance_values = [f[1] for f in sorted_features]
        
        # Plot horizontal bar chart
        plt.barh(range(len(feature_names)), importance_values, align='center')
        plt.yticks(range(len(feature_names)), feature_names)
        
    elif isinstance(importance_data, pd.DataFrame) and 'feature' in importance_data.columns:
        # Get the latest date data
        latest_date = importance_data['date'].max()
        latest_data = importance_data[importance_data['date'] == latest_date]
        
        # Get top N features
        top_features = latest_data.nlargest(top_n, 'importance')
        
        # Plot using seaborn for better appearance
        sns.barplot(x='importance', y='feature', data=top_features.sort_values('importance', ascending=False))
    
    else:
        logging.error("Invalid importance data format")
        return
    
    plt.title(f'Top {top_n} Most Important Features')
    plt.xlabel('Importance')
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logging.info(f"Saved plot to {output_file}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()

def plot_feature_importance_history(history_df, top_n=10, output_file=None, show_plot=True):
    """
    Plot the historical importance of top features over time
    
    Args:
        history_df (pd.DataFrame): DataFrame with historical feature importance
        top_n (int): Number of top features to display
        output_file (str, optional): Path to save the plot
        show_plot (bool): Whether to display the plot
    """
    if history_df.empty:
        logging.error("No historical data available")
        return
    
    # Get top N features across all time points
    overall_top_features = history_df.groupby('feature')['importance'].mean().nlargest(top_n).index.tolist()
    
    # Filter data to only include these features
    plot_data = history_df[history_df['feature'].isin(overall_top_features)]
    
    plt.figure(figsize=(14, 8))
    
    # Plot time series for each feature
    pivot_data = plot_data.pivot(index='date', columns='feature', values='importance')
    pivot_data.plot(marker='o', ax=plt.gca())
    
    plt.title(f'Feature Importance History (Top {top_n} Features)')
    plt.xlabel('Date')
    plt.ylabel('Importance')
    plt.grid(True, alpha=0.3)
    
    # Format x-axis as dates
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    
    # Add legend outside the plot area
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logging.info(f"Saved history plot to {output_file}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()

def plot_feature_category_importance(importance_data, output_file=None, show_plot=True):
    """
    Plot importance by feature category (trend, matchup, rest, etc.)
    
    Args:
        importance_data (dict or pd.DataFrame): Feature importance data
        output_file (str, optional): Path to save the plot
        show_plot (bool): Whether to display the plot
    """
    # Define categories and their patterns
    categories = {
        'Trend Features': ['last3_', 'last10_', '_trend'],
        'Rest Features': ['days_rest', 'back_to_back', 'normal_rest', 'long_rest', 'fatigue'],
        'Matchup Features': ['home_game', 'opponent', 'is_home'],
        'Consistency Features': ['_consistency', 'overall_consistency'],
        'Usage Features': ['usage', 'USG%', 'high_usage', 'med_usage', 'low_usage'],
        'Opponent Strength': ['opp_strength', 'opp_Net_Rating', 'opp_Offensive_Rating'],
        'Player Profile': ['Age', 'MP', 'BPM', 'OBPM', 'DBPM', 'WS/48'],
        'Season Averages': ['PTS', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'FG%', '3P%', 'FT%']
    }
    
    # Convert data to dictionary if needed
    if isinstance(importance_data, pd.DataFrame):
        # Get the latest date data
        if 'date' in importance_data.columns:
            latest_date = importance_data['date'].max()
            latest_data = importance_data[importance_data['date'] == latest_date]
        else:
            latest_data = importance_data
            
        # Convert to dictionary
        importance_dict = dict(zip(latest_data['feature'], latest_data['importance']))
    else:
        importance_dict = importance_data
    
    # Calculate category importance
    category_importance = {}
    uncategorized_importance = 0
    
    for feature, importance in importance_dict.items():
        categorized = False
        for category, patterns in categories.items():
            if any(pattern in feature for pattern in patterns):
                if category not in category_importance:
                    category_importance[category] = 0
                category_importance[category] += importance
                categorized = True
                break
        
        if not categorized:
            uncategorized_importance += importance
    
    # Add uncategorized if there are any
    if uncategorized_importance > 0:
        category_importance['Other'] = uncategorized_importance
    
    # Create pie chart
    plt.figure(figsize=(12, 8))
    
    # Sort categories by importance
    sorted_categories = sorted(category_importance.items(), key=lambda x: x[1], reverse=True)
    labels = [cat[0] for cat in sorted_categories]
    sizes = [cat[1] for cat in sorted_categories]
    
    # Plot with percentage labels
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True, startangle=140)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    
    plt.title('Feature Importance by Category')
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logging.info(f"Saved category plot to {output_file}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()

def create_all_visualizations(importance_file=None, history_dir=None, output_dir=None, show_plots=False):
    """
    Create all feature importance visualizations
    
    Args:
        importance_file (str, optional): Path to feature importance JSON file
        history_dir (str, optional): Directory to search for historical importance files
        output_dir (str, optional): Directory to save visualization files
        show_plots (bool): Whether to display the plots
    """
    # Set default output dir if not provided
    if output_dir is None:
        output_dir = VIZ_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d")
    
    # Load current feature importance
    importance_data = load_feature_importance(importance_file)
    
    if not importance_data:
        logging.error("No feature importance data available")
        return False
    
    # Plot top features
    top_features_file = os.path.join(output_dir, f"top_features_{timestamp}.png")
    plot_top_features(importance_data, top_n=20, output_file=top_features_file, show_plot=show_plots)
    
    # Plot feature categories
    categories_file = os.path.join(output_dir, f"feature_categories_{timestamp}.png")
    plot_feature_category_importance(importance_data, output_file=categories_file, show_plot=show_plots)
    
    # Load and plot historical data if available
    history_df = load_historical_importance(history_dir)
    if not history_df.empty:
        history_file = os.path.join(output_dir, f"feature_history_{timestamp}.png")
        plot_feature_importance_history(history_df, top_n=10, output_file=history_file, show_plot=show_plots)
    
    logging.info(f"All visualizations created and saved to {output_dir}")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Feature Importance Visualization Tool')
    
    # Input options
    parser.add_argument('--importance-file', type=str, 
                      help='Path to feature importance JSON file')
    parser.add_argument('--history-dir', type=str, 
                      help='Directory to search for historical importance files')
    
    # Output options
    parser.add_argument('--output-dir', type=str, 
                      help='Directory to save visualization files')
    parser.add_argument('--show-plots', action='store_true', 
                      help='Display plots (not just save to files)')
    
    # Visualization options
    parser.add_argument('--top-n', type=int, default=20, 
                      help='Number of top features to display')
    
    # Specific plot options
    parser.add_argument('--plot-top', action='store_true', 
                      help='Plot top N most important features')
    parser.add_argument('--plot-history', action='store_true', 
                      help='Plot historical feature importance')
    parser.add_argument('--plot-categories', action='store_true', 
                      help='Plot importance by feature category')
    parser.add_argument('--all-plots', action='store_true', 
                      help='Generate all plots')
    
    args = parser.parse_args()
    
    # Check if at least one plot type was specified
    if not any([args.plot_top, args.plot_history, args.plot_categories, args.all_plots]):
        args.all_plots = True  # Default to all plots if none specified
    
    # Load importance data
    importance_data = load_feature_importance(args.importance_file)
    
    if args.all_plots:
        create_all_visualizations(
            importance_file=args.importance_file,
            history_dir=args.history_dir,
            output_dir=args.output_dir,
            show_plots=args.show_plots
        )
    else:
        # Set default output directory
        output_dir = args.output_dir or VIZ_OUTPUT_DIR
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate timestamp for filenames
        timestamp = datetime.now().strftime("%Y%m%d")
        
        if args.plot_top:
            top_features_file = os.path.join(output_dir, f"top_features_{timestamp}.png")
            plot_top_features(importance_data, top_n=args.top_n, 
                             output_file=top_features_file, show_plot=args.show_plots)
        
        if args.plot_categories:
            categories_file = os.path.join(output_dir, f"feature_categories_{timestamp}.png")
            plot_feature_category_importance(importance_data, 
                                           output_file=categories_file, show_plot=args.show_plots)
        
        if args.plot_history:
            history_df = load_historical_importance(args.history_dir)
            if not history_df.empty:
                history_file = os.path.join(output_dir, f"feature_history_{timestamp}.png")
                plot_feature_importance_history(history_df, top_n=args.top_n, 
                                              output_file=history_file, show_plot=args.show_plots)