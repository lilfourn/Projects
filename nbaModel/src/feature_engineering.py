import pandas as pd
import numpy as np
import re
import os
import json
import glob
from datetime import datetime, timedelta
import logging
import time
from functools import wraps
import pickle

# Try to import data quality module
try:
    from src.data_quality import fix_invalid_values, resolve_turnover_columns
    QUALITY_CHECKS_AVAILABLE = True
except ImportError:
    QUALITY_CHECKS_AVAILABLE = False
    logging.warning("Data quality module not available for feature engineering")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Import from config if available, otherwise set defaults
try:
    from config import (
        get_engineered_data_path, ENGINEERED_DATA_DIR,
        DATA_DIR, ensure_directories_exist
    )
except ImportError:
    # Fallback defaults if config module not available
    DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    ENGINEERED_DATA_DIR = os.path.join(DATA_DIR, "engineered")
    
    def ensure_directories_exist():
        """Ensure all data directories exist"""
        os.makedirs(ENGINEERED_DATA_DIR, exist_ok=True)
    
    def get_engineered_data_path(date=None):
        """Get the path to engineered data file for a specific date"""
        date_str = date or datetime.now().strftime("%Y%m%d")
        return os.path.join(ENGINEERED_DATA_DIR, f"engineered_nba_data_{date_str}.csv")

# Cache directory for intermediate results
CACHE_DIR = os.path.join(ENGINEERED_DATA_DIR, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# Create a file to store feature importance
FEATURE_IMPORTANCE_FILE = os.path.join(ENGINEERED_DATA_DIR, "feature_importance.json")

def cleanup_cache(max_age_days=7, dry_run=False):
    """
    Clean up old cache files
    
    Args:
        max_age_days (int): Maximum age in days to keep cache files
        dry_run (bool): Whether to print what would be deleted without deleting
        
    Returns:
        int: Number of files deleted
    """
    if not os.path.exists(CACHE_DIR):
        return 0
        
    # Get current time
    now = time.time()
    
    # Find all .pkl files in the cache directory
    cache_files = glob.glob(os.path.join(CACHE_DIR, "*.pkl"))
    
    # Count deleted files
    deleted_count = 0
    
    # Check each file's age
    for cache_file in cache_files:
        # Get file modification time
        mod_time = os.path.getmtime(cache_file)
        
        # Calculate age in days
        age_days = (now - mod_time) / (60 * 60 * 24)
        
        # Delete if older than max_age_days
        if age_days > max_age_days:
            if dry_run:
                logging.info(f"Would delete cache file: {os.path.basename(cache_file)} (age: {age_days:.1f} days)")
                deleted_count += 1
            else:
                try:
                    os.remove(cache_file)
                    logging.info(f"Deleted cache file: {os.path.basename(cache_file)} (age: {age_days:.1f} days)")
                    deleted_count += 1
                except Exception as e:
                    logging.warning(f"Error deleting cache file {cache_file}: {str(e)}")
    
    return deleted_count

# Dictionary of feature importance (populated from file if exists)
FEATURE_IMPORTANCE = {}
if os.path.exists(FEATURE_IMPORTANCE_FILE):
    try:
        with open(FEATURE_IMPORTANCE_FILE, 'r') as f:
            FEATURE_IMPORTANCE = json.load(f)
    except:
        pass

# Function timing decorator for performance monitoring
def time_function(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        logging.info(f"Function {func.__name__} took {elapsed_time:.2f} seconds to execute")
        return result
    return wrapper

# Cache decorator for expensive functions
def cache_result(cache_key, expire_days=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Check if caching is explicitly disabled with _use_cache=False
            use_cache = kwargs.pop('_use_cache', True)
            
            # Create cache filename
            cache_file = os.path.join(CACHE_DIR, f"{cache_key}.pkl")
            
            # Check if cache exists and is fresh and we're allowed to use it
            if use_cache and os.path.exists(cache_file):
                # Check if cache has expired
                mod_time = os.path.getmtime(cache_file)
                age_days = (time.time() - mod_time) / (60 * 60 * 24)
                
                if age_days < expire_days:
                    try:
                        with open(cache_file, 'rb') as f:
                            logging.info(f"Loading cached result for {func.__name__} from {cache_file}")
                            return pickle.load(f)
                    except Exception as e:
                        logging.warning(f"Error loading cache: {str(e)}")
            
            # Execute function if no cache or cache expired or caching disabled
            result = func(*args, **kwargs)
            
            # Save result to cache if caching is enabled
            if use_cache:
                try:
                    with open(cache_file, 'wb') as f:
                        logging.info(f"Caching result for {func.__name__} to {cache_file}")
                        pickle.dump(result, f)
                except Exception as e:
                    logging.warning(f"Error writing cache: {str(e)}")
                
            return result
        return wrapper
    return decorator

# Feature importance helpers
def load_feature_importance():
    """Load feature importance from file"""
    if os.path.exists(FEATURE_IMPORTANCE_FILE):
        try:
            with open(FEATURE_IMPORTANCE_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_feature_importance(importance_dict):
    """Save feature importance to file"""
    try:
        with open(FEATURE_IMPORTANCE_FILE, 'w') as f:
            json.dump(importance_dict, f, indent=2)
    except Exception as e:
        logging.warning(f"Error saving feature importance: {str(e)}")

def get_important_features(threshold=0.01):
    """Get list of important features above threshold"""
    importance = load_feature_importance()
    if not importance:
        # If no importance data, return everything
        return None
        
    # Filter features by importance
    important_features = [f for f, score in importance.items() if score >= threshold]
    return important_features if important_features else None

def update_feature_importance_from_model(model_file, feature_names=None, threshold=0.01):
    """
    Extract feature importance from a trained model and update the importance file
    
    Args:
        model_file (str): Path to the trained model file
        feature_names (list, optional): Names of features if different from model's feature_names_
        threshold (float): Minimum importance threshold to keep
        
    Returns:
        dict: Updated feature importance dictionary
    """
    try:
        # Load model
        import joblib
        model = joblib.load(model_file)
        
        # Extract feature importance
        importances = {}
        
        # Handle different model types
        if hasattr(model, 'feature_importances_'):
            # Single tree-based model
            feature_imps = model.feature_importances_
            feat_names = feature_names or getattr(model, 'feature_names_in_', None)
            
            if feat_names is not None and len(feat_names) == len(feature_imps):
                for name, imp in zip(feat_names, feature_imps):
                    importances[name] = float(imp)
                    
        elif hasattr(model, 'estimators_') and len(model.estimators_) > 0:
            # MultiOutputRegressor or ensemble
            if hasattr(model.estimators_[0], 'feature_importances_'):
                # Average feature importance across estimators
                num_estimators = len(model.estimators_)
                all_importances = {}
                
                for i, est in enumerate(model.estimators_):
                    est_importances = est.feature_importances_
                    feat_names = feature_names or getattr(est, 'feature_names_in_', None)
                    
                    if feat_names is not None and len(feat_names) == len(est_importances):
                        for name, imp in zip(feat_names, est_importances):
                            if name not in all_importances:
                                all_importances[name] = []
                            all_importances[name].append(float(imp))
                
                # Calculate average importance
                for name, imps in all_importances.items():
                    importances[name] = sum(imps) / len(imps)
        
        # If we have importances, update the importance file
        if importances:
            # Filter out features with importance below threshold
            importances = {k: v for k, v in importances.items() if v >= threshold}
            
            # Load existing importance data
            existing = load_feature_importance()
            
            # Merge with existing, updating values
            merged = {**existing, **importances}
            
            # Save the updated importance data
            save_feature_importance(merged)
            logging.info(f"Updated feature importance with {len(importances)} features")
            
            return merged
        else:
            logging.warning("Could not extract feature importance from model")
    
    except Exception as e:
        logging.error(f"Error updating feature importance: {str(e)}")
    
    return None

@time_function
@cache_result('matchup_features')
def create_matchup_features(game_data):
    """
    Create features based on the matchup between teams - vectorized version
    
    Args:
        game_data (pd.DataFrame): DataFrame containing game data
        
    Returns:
        pd.DataFrame: DataFrame with additional matchup features
    """
    # Make a copy of the input DataFrame
    df = game_data.copy()
    
    # Convert gameID and teamAbv to strings to ensure consistent processing
    df['gameID_str'] = df['gameID'].astype(str)
    df['teamAbv_str'] = df['teamAbv'].astype(str)
    
    # Create pattern for matching
    df['team_pattern'] = '@' + df['teamAbv_str']
    
    # Vectorized home/away calculation - avoid using ~ operator directly
    # Instead, use == False to invert the boolean Series
    df['is_home'] = (df['gameID_str'].str.contains(df['team_pattern'], regex=False) == False)
    df['home_game'] = df['is_home'].astype(int)
    
    # Extract opponent team using vectorized string operations
    # First, split gameID on '@' character
    df['game_parts'] = df['gameID_str'].str.split('@')
    
    # For home games (is_home==True), opponent is after the @
    # For away games (is_home==False), opponent is before the @ (after removing the date)
    
    # Create a Series to hold all opponents
    opponents = pd.Series(index=df.index, dtype='object')
    
    # Process home games
    home_mask = df['is_home']
    home_indices = df[home_mask].index
    if len(home_indices) > 0:
        home_opponents = df.loc[home_mask, 'game_parts'].str[1]
        opponents.loc[home_indices] = home_opponents.values
    
    # Process away games
    away_mask = ~df['is_home']
    away_indices = df[away_mask].index
    if len(away_indices) > 0:
        away_opponents = df.loc[away_mask, 'game_parts'].str[0].str.split('_').str[1]
        opponents.loc[away_indices] = away_opponents.values
    
    # Assign opponents to the dataframe
    df['opponent'] = opponents
    
    # Clean up opponent (remove date part if present)
    df['opponent'] = df['opponent'].str.replace(r'^\d+_', '', regex=True)
    
    # Drop temporary columns
    df = df.drop(columns=['gameID_str', 'teamAbv_str', 'team_pattern', 'game_parts'])
    
    logging.info(f"Created matchup features with {len(df)} rows")
    return df

@time_function
@cache_result('defensive_matchup_features')
def create_defensive_matchup_features(game_data, team_ratings_path=None):
    """
    Create features based on defensive matchups between players and teams
    
    Args:
        game_data (pd.DataFrame): DataFrame containing game data
        team_ratings_path (str, optional): Path to team ratings file for defensive metrics
        
    Returns:
        pd.DataFrame: DataFrame with additional defensive matchup features
    """
    # Make a copy of the input DataFrame
    df = game_data.copy()
    
    logging.info("Creating defensive matchup features...")
    
    # Load team defensive ratings if available
    team_def_ratings = {}
    if team_ratings_path and os.path.exists(team_ratings_path):
        try:
            team_ratings = pd.read_csv(team_ratings_path)
            # Extract defensive ratings if available
            if 'defensive_rating' in team_ratings.columns and 'team_abbrev' in team_ratings.columns:
                for _, row in team_ratings.iterrows():
                    if not pd.isna(row['team_abbrev']):
                        team_def_ratings[row['team_abbrev']] = row['defensive_rating']
            logging.info(f"Loaded defensive ratings for {len(team_def_ratings)} teams")
        except Exception as e:
            logging.error(f"Error loading team defensive ratings: {str(e)}")
    
    # Create opponent defensive rating feature
    if team_def_ratings:
        df['opp_defensive_rating'] = df['opponent'].map(team_def_ratings)
        
        # Fill missing values with median
        median_rating = pd.Series(team_def_ratings.values()).median()
        df['opp_defensive_rating'] = df['opp_defensive_rating'].fillna(median_rating)
        
        # Normalize to a 0-1 scale (higher is better defense)
        min_rating = df['opp_defensive_rating'].min()
        max_rating = df['opp_defensive_rating'].max()
        if max_rating > min_rating:
            df['opp_def_strength'] = (df['opp_defensive_rating'] - min_rating) / (max_rating - min_rating)
        else:
            df['opp_def_strength'] = 0.5
    else:
        # If no ratings data available, create placeholder based on historical data
        logging.info("No team ratings available, creating placeholder defensive metrics")
        # Group by opponent and calculate average points allowed
        opp_def = df.groupby('opponent')['pts'].mean().reset_index()
        opp_def = opp_def.rename(columns={'pts': 'avg_pts_allowed'})
        
        # Merge back to main dataframe
        df = pd.merge(df, opp_def, on='opponent', how='left')
        
        # Convert to a defensive strength metric (inverted - lower points is better defense)
        max_pts = df['avg_pts_allowed'].max()
        min_pts = df['avg_pts_allowed'].min()
        if max_pts > min_pts:
            df['opp_def_strength'] = 1 - ((df['avg_pts_allowed'] - min_pts) / (max_pts - min_pts))
        else:
            df['opp_def_strength'] = 0.5
            
        # Drop intermediate column
        df = df.drop(columns=['avg_pts_allowed'])
    
    # Create position-specific defensive matchup features
    if 'position' in df.columns:
        # Group by opponent and position to see how teams defend each position
        pos_groups = df.groupby(['opponent', 'position'])
        
        for stat in ['pts', 'reb', 'ast', 'fgm', 'tptfgm']:
            if stat not in df.columns:
                continue
                
            # Calculate average stat by position for each opponent
            pos_stat = pos_groups[stat].mean().reset_index()
            pos_stat = pos_stat.rename(columns={stat: f'opp_vs_{stat}_by_pos'})
            
            # Get league average for each position
            league_pos_avg = df.groupby('position')[stat].mean().reset_index()
            league_pos_avg = league_pos_avg.rename(columns={stat: f'league_avg_{stat}_by_pos'})
            
            # Merge position-specific averages
            df = pd.merge(df, pos_stat, on=['opponent', 'position'], how='left')
            df = pd.merge(df, league_pos_avg, on=['position'], how='left')
            
            # Calculate defensive matchup advantage (>1 means team defends position poorly)
            df[f'def_matchup_{stat}'] = df[f'opp_vs_{stat}_by_pos'] / df[f'league_avg_{stat}_by_pos']
            
            # Fill missing values
            df[f'def_matchup_{stat}'] = df[f'def_matchup_{stat}'].fillna(1.0)
            
            # Drop intermediate columns
            df = df.drop(columns=[f'opp_vs_{stat}_by_pos', f'league_avg_{stat}_by_pos'])
    
    # Create defensive efficiency metrics based on shot location if available
    shot_loc_cols = [col for col in df.columns if 'zone' in col.lower() and 'pct' in col.lower()]
    if shot_loc_cols:
        # Create metrics for teams that defend specific areas well
        logging.info("Creating shot location defensive features")
        # Implementation would depend on available data
    
    # Create features for recent defensive performance trends
    # Group by opponent and get recent games
    if 'game_date' in df.columns:
        # Create a copy of the dataframe with just the needed columns to reduce memory use
        temp_df = df[['opponent', 'game_date', 'pts']].copy()
        
        # Calculate recent defensive performance for each team (last 10 games)
        for team in df['opponent'].unique():
            team_games = temp_df[temp_df['opponent'] == team].sort_values('game_date')
            
            # Calculate rolling average of points allowed
            team_games['recent_def_pts'] = team_games['pts'].rolling(window=10, min_periods=3).mean()
            
            # Merge back to main dataframe
            recent_def = team_games[['game_date', 'recent_def_pts']].dropna()
            
            # For each game with this opponent, find the most recent defensive metrics
            for date in df[df['opponent'] == team]['game_date'].unique():
                prior_games = recent_def[recent_def['game_date'] < date]
                
                if not prior_games.empty:
                    most_recent = prior_games.iloc[-1]['recent_def_pts']
                    df.loc[(df['opponent'] == team) & (df['game_date'] == date), 'opp_recent_def'] = most_recent
        
        # Normalize recent defensive metrics
        if 'opp_recent_def' in df.columns:
            min_val = df['opp_recent_def'].min()
            max_val = df['opp_recent_def'].max()
            if max_val > min_val:
                df['opp_recent_def_strength'] = 1 - ((df['opp_recent_def'] - min_val) / (max_val - min_val))
            
            # Drop intermediate column
            df = df.drop(columns=['opp_recent_def'])
    
    # Fill any missing values
    def_cols = [col for col in df.columns if 'def_' in col or 'opp_' in col]
    for col in def_cols:
        df[col] = df[col].fillna(0.5)  # Neutral value for missing data
    
    logging.info(f"Created {len(def_cols)} defensive matchup features with {len(df)} rows")
    return df

@time_function
@cache_result('rest_features')
def create_rest_features(game_data):
    """
    Create features based on rest days between games - vectorized version
    
    Args:
        game_data (pd.DataFrame): DataFrame containing game data with dates
        
    Returns:
        pd.DataFrame: DataFrame with additional rest features
    """
    # Make a copy of the input DataFrame
    df = game_data.copy()
    
    # Ensure game_date is datetime
    if 'game_date' not in df.columns:
        # Extract date from gameID (format: YYYYMMDD_TEAM@TEAM)
        df['game_date'] = df['gameID'].astype(str).str.split('_', expand=True)[0]
    
    # Convert to datetime, with error handling
    df['game_date'] = pd.to_datetime(df['game_date'], format='%Y%m%d', errors='coerce')
    
    # Sort by player and date
    df = df.sort_values(['playerID', 'game_date'])
    
    # Calculate days since last game (rest days) - vectorized
    df['prev_game_date'] = df.groupby('playerID')['game_date'].shift(1)
    df['days_rest'] = (df['game_date'] - df['prev_game_date']).dt.days
    
    # Fill NaN values (first game of the season) with median rest days
    median_rest = df['days_rest'].median()
    df['days_rest'] = df['days_rest'].fillna(median_rest)
    
    # Create enhanced rest features (all vectorized)
    
    # Rest categories
    df['back_to_back'] = (df['days_rest'] < 2).astype(int)
    df['normal_rest'] = ((df['days_rest'] >= 2) & (df['days_rest'] <= 3)).astype(int)
    df['long_rest'] = (df['days_rest'] > 3).astype(int)
    df['very_long_rest'] = (df['days_rest'] > 7).astype(int)
    
    # Rest impact scores using numpy's select function (vectorized)
    conditions = [
        df['days_rest'] < 2,                           # Back-to-back
        (df['days_rest'] >= 2) & (df['days_rest'] <= 3),  # Optimal rest
        (df['days_rest'] > 3) & (df['days_rest'] <= 6),   # Longer rest
        df['days_rest'] > 6                            # Very long rest
    ]
    
    choices = [-0.2, 0.1, 0.0, -0.1]
    df['rest_impact'] = np.select(conditions, choices, default=0)
    
    # Back-to-back impact specific features (all vectorized)
    df['second_game_b2b'] = ((df['back_to_back'] == 1) & 
                              (df.groupby('playerID')['back_to_back'].shift(1) == 1)).astype(int)
    
    df['third_game_in_4_days'] = ((df['days_rest'] < 2) & 
                                  (df.groupby('playerID')['days_rest'].shift(1) < 3)).astype(int)
    
    # Calculate games in last 7 days (simplified approach for robustness)
    df['is_game'] = 1
    df['games_last_7_days'] = 0  # Initialize with zeros
    
    # Process each player group separately
    for player_id, player_df in df.groupby('playerID'):
        if len(player_df) <= 1:
            continue
            
        # Make sure dates are sorted
        player_df = player_df.sort_values('game_date')
        
        # Remove rows with NaT dates
        valid_dates = player_df['game_date'].notna()
        if not valid_dates.all():
            player_df = player_df[valid_dates].copy()
            
        if len(player_df) <= 1:
            continue
            
        # For each game, count games in previous 7 days
        for i, (idx, row) in enumerate(player_df.iterrows()):
            if i == 0:  # First game has 0 previous games
                continue
                
            # Get the date to look back from
            current_date = row['game_date']
            if pd.isna(current_date):
                continue
                
            # Define the cutoff date (7 days before)
            cutoff_date = current_date - pd.Timedelta(days=7)
            
            # Count games in the 7-day window
            prev_games = player_df[
                (player_df['game_date'] >= cutoff_date) & 
                (player_df['game_date'] < current_date)
            ]
            count = len(prev_games)
            
            # Update the count in the main dataframe
            df.loc[idx, 'games_last_7_days'] = count
    
    # Fill NaNs with 0 for first games
    df['games_last_7_days'] = df['games_last_7_days'].fillna(0)
    
    # Create fatigue score using vectorized numpy select
    fatigue_conditions = [
        df['games_last_7_days'] <= 2,  # Low fatigue
        df['games_last_7_days'] == 3,  # Moderate fatigue
        df['games_last_7_days'] == 4,  # High fatigue
        df['games_last_7_days'] >= 5   # Very high fatigue
    ]
    
    fatigue_choices = [0, 0.3, 0.6, 1.0]
    df['fatigue_score'] = np.select(fatigue_conditions, fatigue_choices, default=0)
    
    # Drop temporary column
    df = df.drop(columns=['is_game'])
    
    logging.info(f"Created rest features with {len(df)} rows")
    return df

@time_function
@cache_result('trend_features')
def create_trend_features(game_data, selective=True, use_weighted_averages=True):
    """
    Create features based on recent performance trends with optional weighted averages
    
    Args:
        game_data (pd.DataFrame): DataFrame containing game data
        selective (bool): Whether to selectively create only important features
        use_weighted_averages (bool): Whether to use weighted averages for recent games
        
    Returns:
        pd.DataFrame: DataFrame with additional trend features
    """
    # Make a copy of the input DataFrame
    df = game_data.copy()
    
    # Sort by player and date
    if 'game_date' not in df.columns:
        # Extract date from gameID (format: YYYYMMDD_TEAM@TEAM)
        df['game_date'] = df['gameID'].astype(str).str.split('_', expand=True)[0]
        df['game_date'] = pd.to_datetime(df['game_date'], format='%Y%m%d', errors='coerce')
    
    df = df.sort_values(['playerID', 'game_date'])
    
    # Get the list of stats to compute trends for
    all_stat_cols = ['pts', 'reb', 'ast', 'stl', 'blk', 'TOV_x', 'fgm', 'fga', 'fgp',
                'tptfgm', 'tptfga', 'tptfgp', 'ftm', 'fta', 'ftp', 
                'mins', 'plusMinus', 'eFG%', 'TS%', 'AST_TO_ratio', 'PPS']
    
    # Only include columns that actually exist in the dataframe
    available_cols = [col for col in all_stat_cols if col in df.columns]
    
    # If selective is True, use feature importance to pick most significant stats
    if selective:
        # Get feature importance from saved data
        importance = load_feature_importance()
        
        # Determine which statistics have high importance in trend features
        # Look for any trend features in the importance data
        trend_pattern = re.compile(r'(last3_|last10_|_trend|_w_avg)')
        trend_importances = {k: v for k, v in importance.items() if trend_pattern.search(k)}
        
        if trend_importances:
            # Extract the base stat names from important trend features
            base_stats = set()
            for feature in trend_importances.keys():
                # Remove prefixes/suffixes to get the base stat name
                base_stat = re.sub(r'last3_|last10_|_w_avg|_avg|_trend', '', feature)
                base_stats.add(base_stat)
            
            # Filter to only important base stats that are available
            stat_cols = [col for col in available_cols if col in base_stats]
            
            # If we don't have any important stats, fall back to primary stats
            if not stat_cols:
                # Just use the primary stats instead of all stats
                stat_cols = ['pts', 'reb', 'ast', 'mins']
                
                # Add advanced metrics if they exist
                for metric in ['eFG%', 'TS%', 'AST_TO_ratio']:
                    if metric in available_cols:
                        stat_cols.append(metric)
                
                stat_cols = [col for col in stat_cols if col in available_cols]
        else:
            # No importance data for trends, use primary stats only
            stat_cols = ['pts', 'reb', 'ast', 'mins']
            
            # Add advanced metrics if they exist
            for metric in ['eFG%', 'TS%', 'AST_TO_ratio']:
                if metric in available_cols:
                    stat_cols.append(metric)
            
            stat_cols = [col for col in stat_cols if col in available_cols]
    else:
        # Use all available columns
        stat_cols = available_cols
    
    # Log which stats we're creating trends for
    logging.info(f"Creating trend features for {len(stat_cols)} stats: {stat_cols}")
    
    # Create all trend features efficiently in a single pass
    # Group by playerID first to avoid multiple groupby operations
    player_groups = df.groupby('playerID')
    
    # Calculate rolling averages for last 3 and last 10 games for each stat
    for stat in stat_cols:
        # Standard unweighted averages
        
        # Last 3 games average (vectorized)
        df[f'last3_{stat}_avg'] = player_groups[stat].rolling(window=3, min_periods=1).mean().reset_index(level=0, drop=True).shift(1)
        
        # Last 10 games average (vectorized)
        df[f'last10_{stat}_avg'] = player_groups[stat].rolling(window=10, min_periods=1).mean().reset_index(level=0, drop=True).shift(1)
        
        # Weighted averages (if enabled)
        if use_weighted_averages:
            # Define a function for exponentially weighted average calculation
            # This gives higher weight to more recent games
            
            # Exponentially weighted moving average for last 3 games (higher weight to recent games)
            df[f'last3_{stat}_w_avg'] = player_groups[stat].ewm(span=3, min_periods=1).mean().reset_index(level=0, drop=True).shift(1)
            
            # Exponentially weighted moving average for last 10 games
            df[f'last10_{stat}_w_avg'] = player_groups[stat].ewm(span=10, min_periods=1).mean().reset_index(level=0, drop=True).shift(1)
            
            # Calculate trends using weighted averages
            df[f'{stat}_w_trend'] = df[f'last3_{stat}_w_avg'] - df[f'last10_{stat}_w_avg']
        
        # Calculate trend (difference between short and medium term)
        df[f'{stat}_trend'] = df[f'last3_{stat}_avg'] - df[f'last10_{stat}_avg']
    
    # Fill all NaN values with 0 (vectorized)
    trend_cols = [col for col in df.columns if (
        col.startswith('last3_') or 
        col.startswith('last10_') or 
        col.endswith('_trend') or
        col.endswith('_w_avg') or
        col.endswith('_w_trend')
    )]
    df[trend_cols] = df[trend_cols].fillna(0)
    
    logging.info(f"Created {len(trend_cols)} trend features with {len(df)} rows")
    return df

@time_function
@cache_result('time_weighted_features')
def create_time_weighted_features(game_data, decay_factor=0.9, max_games=20):
    """
    Create features that give higher weight to more recent performances
    
    Args:
        game_data (pd.DataFrame): DataFrame containing game data
        decay_factor (float): Factor to decay weights for older games (0-1)
        max_games (int): Maximum number of past games to include
        
    Returns:
        pd.DataFrame: DataFrame with additional time-weighted features
    """
    # Make a copy of the input DataFrame
    df = game_data.copy()
    
    logging.info("Creating time-weighted features...")
    
    # Sort by player and date
    if 'game_date' not in df.columns:
        # Try to extract date from gameID
        if 'gameID' in df.columns:
            df['game_date'] = pd.to_datetime(
                df['gameID'].astype(str).str.split('_', expand=True)[0], 
                format='%Y%m%d', 
                errors='coerce'
            )
        else:
            logging.warning("Cannot create time-weighted features: missing date information")
            return df
    
    # Get the list of stats to compute time-weighted features for
    stat_cols = ['pts', 'reb', 'ast', 'stl', 'blk', 'fgm', 'fga', 'tptfgm', 'tptfga', 'mins']
    
    # Only include columns that actually exist in the dataframe
    available_stats = [col for col in stat_cols if col in df.columns]
    
    # Initialize feature tracking
    time_weighted_cols = []
    
    # Create a DataFrame to store results to avoid repeated column creation
    result_df = pd.DataFrame(index=df.index)
    
    # Process each player separately to apply time-weighted calculations
    for player_id, player_df in df.groupby('playerID'):
        # Skip if player has too few games
        if len(player_df) < 3:
            continue
            
        # Sort by date
        player_df = player_df.sort_values('game_date')
        player_dates = player_df['game_date'].values
        
        # Process each game for this player
        for idx, row in player_df.iterrows():
            current_date = row['game_date']
            
            # Get all games before current game
            past_games = player_df[player_df['game_date'] < current_date]
            
            # Skip if not enough past games
            if len(past_games) < 3:
                continue
                
            # Limit to max_games most recent games
            past_games = past_games.sort_values('game_date', ascending=False).head(max_games)
            
            # Create weights based on recency
            # Calculate days between current game and past games
            days_diff = [(current_date - date).days for date in past_games['game_date']]
            
            # Apply exponential decay based on days difference
            # More recent games get higher weights
            weights = np.array([decay_factor ** (d/7) for d in days_diff])  # Decay by week
            
            # Normalize weights to sum to 1
            weights = weights / weights.sum() if weights.sum() > 0 else np.ones(len(weights)) / len(weights)
            
            # Calculate weighted averages for each stat
            for stat in available_stats:
                if stat not in past_games.columns:
                    continue
                    
                # Get past values
                past_values = past_games[stat].values
                
                # Calculate weighted average
                weighted_avg = np.sum(past_values * weights)
                
                # Create column name
                col_name = f'tw_{stat}'
                
                # Add to results
                result_df.loc[idx, col_name] = weighted_avg
                
                # Track columns created
                if col_name not in time_weighted_cols:
                    time_weighted_cols.append(col_name)
            
            # Create exponential moving average features (EMA)
            # These give higher weight to recent games but with smoother transitions
            for stat in available_stats:
                if stat not in past_games.columns:
                    continue
                
                # Use pandas built-in EMA calculation
                # Try different alpha values (smoothing factors)
                for alpha in [0.3, 0.7]:  # 0.3 = more smoothing, 0.7 = more responsive
                    # Create temp series with past values plus current value
                    temp_series = pd.Series(
                        np.append(past_games[stat].values, [row[stat]]), 
                        index=np.append(past_games['game_date'].values, [current_date])
                    )
                    
                    # Calculate EMA - drop the last value (current game)
                    ema_values = temp_series.ewm(alpha=alpha).mean().iloc[:-1].values
                    
                    # Only use if we have enough values
                    if len(ema_values) > 0:
                        # Column name encodes the alpha value
                        alpha_str = str(int(alpha * 10))
                        col_name = f'ema{alpha_str}_{stat}'
                        
                        # Store the last EMA value (most recent)
                        result_df.loc[idx, col_name] = ema_values[-1]
                        
                        # Track columns created
                        if col_name not in time_weighted_cols:
                            time_weighted_cols.append(col_name)
            
            # Create momentum indicators (direction of recent performance)
            for stat in ['pts', 'reb', 'ast']:
                if stat not in past_games.columns:
                    continue
                
                # Need at least 5 games to calculate meaningful momentum
                if len(past_games) >= 5:
                    # Recent 3 games average
                    recent_3 = past_games.sort_values('game_date', ascending=False).head(3)[stat].mean()
                    
                    # Previous 5 games average (excluding most recent 3)
                    prev_games = past_games.sort_values('game_date', ascending=False)
                    if len(prev_games) > 3:
                        prev_5 = prev_games.iloc[3:8][stat].mean() if len(prev_games) >= 8 else prev_games.iloc[3:][stat].mean()
                        
                        # Calculate momentum (ratio of recent to previous performance)
                        momentum = recent_3 / prev_5 if prev_5 > 0 else 1.0
                        
                        col_name = f'momentum_{stat}'
                        result_df.loc[idx, col_name] = momentum
                        
                        # Add interpretation column - categorical momentum indicator
                        col_name_cat = f'momentum_{stat}_cat'
                        if momentum > 1.15:
                            result_df.loc[idx, col_name_cat] = 2  # Strong positive momentum
                        elif momentum > 1.05:
                            result_df.loc[idx, col_name_cat] = 1  # Positive momentum
                        elif momentum < 0.85:
                            result_df.loc[idx, col_name_cat] = -2  # Strong negative momentum
                        elif momentum < 0.95:
                            result_df.loc[idx, col_name_cat] = -1  # Negative momentum
                        else:
                            result_df.loc[idx, col_name_cat] = 0  # Stable performance
                        
                        # Track columns
                        if col_name not in time_weighted_cols:
                            time_weighted_cols.append(col_name)
                        if col_name_cat not in time_weighted_cols:
                            time_weighted_cols.append(col_name_cat)
    
    # Merge results back to main dataframe
    df = pd.concat([df, result_df], axis=1)
    
    # Fill NaN values
    for col in time_weighted_cols:
        if col in df.columns:
            # For categorical columns, fill with 0 (neutral)
            if 'cat' in col:
                df[col] = df[col].fillna(0)
            else:
                # For continuous columns, use median of that column
                df[col] = df[col].fillna(df[col].median())
    
    logging.info(f"Created {len(time_weighted_cols)} time-weighted features with {len(df)} rows")
    return df

def create_opp_strength_features(game_data, team_ratings_path=None):
    """
    Create features based on opponent strength using team ratings
    
    Args:
        game_data (pd.DataFrame): DataFrame containing game data
        team_ratings_path (str, optional): Path to the team ratings file
        
    Returns:
        pd.DataFrame: DataFrame with additional opponent strength features
    """
    # Make a copy of the input DataFrame
    df = game_data.copy()
    
    if team_ratings_path is None:
        # Try to find the latest team ratings file
        latest_date = datetime.now().strftime("%Y%m%d")
        team_ratings_path = f"/Users/lukesmac/Projects/nbaModel/data/standings/team_ratings_{latest_date}.csv"
        
        # If the file doesn't exist, use a placeholder approach
        if not pd.io.common.file_exists(team_ratings_path):
            logging.warning(f"Team ratings file not found: {team_ratings_path}")
            
            # Create simple opponent strength categories based on historic team performance
            strong_teams = ['BOS', 'DEN', 'LAL', 'MIL', 'PHI', 'GSW', 'MIA']
            medium_teams = ['DAL', 'NYK', 'LAC', 'PHO', 'CLE', 'MEM', 'SAC', 'NOP', 'MIN']
            weak_teams = ['OKC', 'ATL', 'BRK', 'TOR', 'CHI', 'WAS', 'UTA', 'POR', 'SAS', 'CHO', 'ORL', 'IND', 'HOU', 'DET']
            
            df['opp_strength'] = 0  # Default medium
            df.loc[df['opponent'].isin(strong_teams), 'opp_strength'] = 1  # Strong
            df.loc[df['opponent'].isin(weak_teams), 'opp_strength'] = -1  # Weak
            
            return df
    
    # Load team ratings data
    try:
        team_ratings = pd.read_csv(team_ratings_path)
        logging.info(f"Loaded team ratings with {len(team_ratings)} teams")
    except Exception as e:
        logging.error(f"Error loading team ratings: {str(e)}")
        # Create a placeholder column
        df['opp_strength'] = 0
        return df
    
    # Create opponent strength features
    
    # Map team names in ratings to team abbreviations
    team_mapping = {
        'Atlanta Hawks': 'ATL',
        'Boston Celtics': 'BOS',
        'Brooklyn Nets': 'BRK',
        'Charlotte Hornets': 'CHO',
        'Chicago Bulls': 'CHI',
        'Cleveland Cavaliers': 'CLE',
        'Dallas Mavericks': 'DAL',
        'Denver Nuggets': 'DEN',
        'Detroit Pistons': 'DET',
        'Golden State Warriors': 'GSW',
        'Houston Rockets': 'HOU',
        'Indiana Pacers': 'IND',
        'Los Angeles Clippers': 'LAC',
        'Los Angeles Lakers': 'LAL',
        'Memphis Grizzlies': 'MEM',
        'Miami Heat': 'MIA',
        'Milwaukee Bucks': 'MIL',
        'Minnesota Timberwolves': 'MIN',
        'New Orleans Pelicans': 'NOP',
        'New York Knicks': 'NYK',
        'Oklahoma City Thunder': 'OKC',
        'Orlando Magic': 'ORL',
        'Philadelphia 76ers': 'PHI',
        'Phoenix Suns': 'PHO',
        'Portland Trail Blazers': 'POR',
        'Sacramento Kings': 'SAC',
        'San Antonio Spurs': 'SAS',
        'Toronto Raptors': 'TOR',
        'Utah Jazz': 'UTA',
        'Washington Wizards': 'WAS'
    }
    
    # Use the mapping to create a new column with team abbreviations
    if 'Team' in team_ratings.columns:
        team_ratings['Team_Abbr'] = team_ratings['Team'].map(team_mapping)
    
    # Extract relevant ratings columns
    rating_cols = ['Offensive_Rating', 'Defensive_Rating', 'Net_Rating', 'W', 'L']
    rating_cols = [col for col in rating_cols if col in team_ratings.columns]
    
    # Create dictionary of team ratings for merging
    team_rating_dict = {}
    for _, row in team_ratings.iterrows():
        if 'Team_Abbr' in row and not pd.isna(row['Team_Abbr']):
            team_abbr = row['Team_Abbr']
            team_rating_dict[team_abbr] = {col: row[col] for col in rating_cols if col in row}
    
    # Add opponent ratings to the game data
    for col in rating_cols:
        df[f'opp_{col}'] = df['opponent'].map(lambda x: team_rating_dict.get(x, {}).get(col, 0))
    
    # Create a normalized opponent strength score based on net rating
    if 'opp_Net_Rating' in df.columns:
        # Scale to 0-1 range
        min_rating = df['opp_Net_Rating'].min()
        max_rating = df['opp_Net_Rating'].max()
        if max_rating > min_rating:
            df['opp_strength'] = (df['opp_Net_Rating'] - min_rating) / (max_rating - min_rating)
        else:
            df['opp_strength'] = 0.5  # Default if all ratings are the same
    else:
        df['opp_strength'] = 0.5  # Default if no net rating column
    
    logging.info(f"Created opponent strength features with {len(df)} rows")
    return df

@time_function
@cache_result('consistency_features')
def create_player_consistency_features(game_data, selective=True):
    """
    Create features based on player consistency in performance
    
    Args:
        game_data (pd.DataFrame): DataFrame containing game data
        selective (bool): Whether to selectively create only important features
        
    Returns:
        pd.DataFrame: DataFrame with additional player consistency features
    """
    # Make a copy of the input DataFrame
    df = game_data.copy()
    
    # Sort by player and date
    if 'game_date' not in df.columns:
        # Extract date from gameID (format: YYYYMMDD_TEAM@TEAM)
        df['game_date'] = df['gameID'].astype(str).str.split('_', expand=True)[0]
        df['game_date'] = pd.to_datetime(df['game_date'], format='%Y%m%d', errors='coerce')
    
    df = df.sort_values(['playerID', 'game_date'])
    
    # All possible stats for measuring consistency
    all_stat_cols = ['pts', 'reb', 'ast', 'mins']
    
    # Only include columns that actually exist in the dataframe
    available_cols = [col for col in all_stat_cols if col in df.columns]
    
    # If selective is True, use feature importance to pick most significant stats
    if selective:
        # Get feature importance from saved data
        importance = load_feature_importance()
        
        # Determine which statistics have high importance in consistency features
        consistency_pattern = re.compile(r'(_consistency)')
        consistency_importances = {k: v for k, v in importance.items() if consistency_pattern.search(k)}
        
        if consistency_importances:
            # Extract the base stat names from important consistency features
            base_stats = set()
            for feature in consistency_importances.keys():
                # Remove suffix to get the base stat name
                base_stat = re.sub(r'_consistency', '', feature)
                base_stats.add(base_stat)
            
            # Filter to only important base stats that are available
            stat_cols = [col for col in available_cols if col in base_stats]
            
            # If we don't have any important stats, fall back to primary stats
            if not stat_cols:
                # Just use points consistency at minimum as it's typically most important
                stat_cols = ['pts'] if 'pts' in available_cols else available_cols[:1]
        else:
            # No importance data, use only points consistency
            stat_cols = ['pts'] if 'pts' in available_cols else available_cols[:1]
    else:
        # Use all available columns
        stat_cols = available_cols
    
    # Log which stats we're creating consistency measures for
    logging.info(f"Creating consistency features for {len(stat_cols)} stats: {stat_cols}")
    
    # Group by playerID to avoid multiple groupby operations
    player_groups = df.groupby('playerID')
    
    # Calculate consistency metrics for each selected stat
    for stat in stat_cols:
        # Calculate rolling mean and std for last 10 games (vectorized)
        roll_mean = player_groups[stat].rolling(window=10, min_periods=5).mean().reset_index(level=0, drop=True).shift(1)
        roll_std = player_groups[stat].rolling(window=10, min_periods=5).std().reset_index(level=0, drop=True).shift(1)
        
        # Calculate CV (with handling for division by zero)
        df[f'{stat}_consistency'] = roll_std / roll_mean.replace(0, np.nan)
        
        # Fill NaNs, then invert the CV so higher values mean more consistency
        df[f'{stat}_consistency'] = df[f'{stat}_consistency'].fillna(0)
        df[f'{stat}_consistency'] = 1 / (1 + df[f'{stat}_consistency'])
    
    # Create an overall consistency score (average of individual consistency scores)
    consistency_cols = [f'{stat}_consistency' for stat in stat_cols]
    if consistency_cols:  # Only if we have any consistency columns
        df['overall_consistency'] = df[consistency_cols].mean(axis=1)
    
    logging.info(f"Created player consistency features with {len(df)} rows")
    return df

def create_advanced_offensive_features(game_data):
    """
    Create advanced offensive efficiency metrics
    
    Args:
        game_data (pd.DataFrame): DataFrame containing game data
        
    Returns:
        pd.DataFrame: DataFrame with additional advanced offensive metrics
    """
    # Make a copy of the input DataFrame
    df = game_data.copy()
    
    # Calculate eFG% (Effective Field Goal %) if not already present
    # Formula: eFG% = (FGM + 0.5 * 3PTM) / FGA
    if 'eFG%' not in df.columns:
        # Check if we have the necessary columns
        if all(col in df.columns for col in ['fgm', 'tptfgm', 'fga']):
            # Handle division by zero
            mask = df['fga'] > 0
            df.loc[mask, 'eFG%'] = (df.loc[mask, 'fgm'] + 0.5 * df.loc[mask, 'tptfgm']) / df.loc[mask, 'fga']
            df['eFG%'] = df['eFG%'].fillna(0)
            logging.info("Calculated eFG% from game data")
    
    # Calculate TS% (True Shooting %) if not already present
    # Formula: TS% = PTS / (2 * (FGA + 0.44 * FTA))
    if 'TS%' not in df.columns:
        # Check if we have the necessary columns
        if all(col in df.columns for col in ['pts', 'fga', 'fta']):
            # Handle division by zero
            denominator = 2 * (df['fga'] + 0.44 * df['fta'])
            mask = denominator > 0
            df.loc[mask, 'TS%'] = df.loc[mask, 'pts'] / denominator[mask]
            df['TS%'] = df['TS%'].fillna(0)
            logging.info("Calculated TS% from game data")
    
    # Calculate Assist-to-Turnover ratio
    if 'AST_TO_ratio' not in df.columns:
        if all(col in df.columns for col in ['ast', 'TOV_x']):
            # Handle division by zero
            mask = df['TOV_x'] > 0
            df.loc[mask, 'AST_TO_ratio'] = df.loc[mask, 'ast'] / df.loc[mask, 'TOV_x']
            # For games with 0 turnovers but positive assists, set a high ratio
            mask_zero_tov = (df['TOV_x'] == 0) & (df['ast'] > 0)
            df.loc[mask_zero_tov, 'AST_TO_ratio'] = df.loc[mask_zero_tov, 'ast'] * 2  # A reasonable high value
            # For games with 0 assists and 0 turnovers, set to 1 (neutral)
            mask_both_zero = (df['TOV_x'] == 0) & (df['ast'] == 0)
            df.loc[mask_both_zero, 'AST_TO_ratio'] = 1
            logging.info("Calculated Assist-to-Turnover ratio")
    
    # Points per Shot (PPS) - another useful efficiency metric
    if 'PPS' not in df.columns:
        if all(col in df.columns for col in ['pts', 'fga']):
            mask = df['fga'] > 0
            df.loc[mask, 'PPS'] = df.loc[mask, 'pts'] / df.loc[mask, 'fga']
            df['PPS'] = df['PPS'].fillna(0)
            logging.info("Calculated Points Per Shot (PPS)")
    
    # Create relative offensive efficiency metrics compared to team or league average
    # (if team averages are available)
    
    logging.info(f"Created advanced offensive features with {len(df)} rows")
    return df

def create_usage_features(game_data):
    """
    Create features based on player usage patterns
    
    Args:
        game_data (pd.DataFrame): DataFrame containing game data and season averages
        
    Returns:
        pd.DataFrame: DataFrame with additional usage features
    """
    # Make a copy of the input DataFrame
    df = game_data.copy()
    
    # Check if we have USG% from season averages
    if 'USG%' in df.columns:
        # No transformation needed, already exists
        pass
    elif 'USG' in df.columns:
        # Rename to standard format
        df['USG%'] = df['USG']
    else:
        # Try to calculate a simplified usage approximation from game data
        if all(col in df.columns for col in ['fga', 'fta', 'TOV_x', 'mins']):
            # Simple formula: (FGA + 0.44*FTA + TOV) / minutes
            df['approx_usg'] = (df['fga'] + 0.44 * df['fta'] + df['TOV_x']) / df['mins'].replace(0, np.nan)
            df['approx_usg'] = df['approx_usg'].fillna(0)
            
            # Scale to a percentage format (0-100)
            max_usg = df['approx_usg'].max()
            if max_usg > 0:
                df['USG%'] = (df['approx_usg'] / max_usg) * 100
            else:
                df['USG%'] = 0
    
    # Create usage-related features
    if 'USG%' in df.columns:
        # Usage categories
        df['high_usage'] = (df['USG%'] > 25).astype(int)
        df['med_usage'] = ((df['USG%'] <= 25) & (df['USG%'] > 15)).astype(int)
        df['low_usage'] = (df['USG%'] <= 15).astype(int)
        
        # Calculate recent usage trend (last 5 games vs season average)
        if 'last5_fga_avg' in df.columns and 'last5_fta_avg' in df.columns and 'last5_TOV_avg' in df.columns:
            # We can estimate recent usage with available stats
            pass
    
    # Create usage-efficiency interaction
    if 'USG%' in df.columns and 'TS%' in df.columns:
        df['usage_efficiency'] = df['USG%'] * df['TS%']
    
    logging.info(f"Created usage features with {len(df)} rows")
    return df

def create_player_specific_factors(game_data):
    """
    Create features based on player-specific factors like injuries, minutes restrictions, and team changes
    
    Args:
        game_data (pd.DataFrame): DataFrame containing game data
        
    Returns:
        pd.DataFrame: DataFrame with additional player-specific features
    """
    # Make a copy of the input DataFrame
    df = game_data.copy()
    
    # Sort by player and date
    if 'game_date' not in df.columns:
        # Extract date from gameID (format: YYYYMMDD_TEAM@TEAM)
        df['game_date'] = df['gameID'].astype(str).str.split('_', expand=True)[0]
        df['game_date'] = pd.to_datetime(df['game_date'], format='%Y%m%d', errors='coerce')
    
    df = df.sort_values(['playerID', 'game_date'])
    
    # Detect potential minutes restrictions by comparing to season averages
    # This is an approximation as we don't have actual injury data
    if 'mins' in df.columns:
        # Calculate each player's median minutes per game
        player_median_mins = df.groupby('playerID')['mins'].median().reset_index()
        player_median_mins.columns = ['playerID', 'median_mins']
        
        # Merge back to the main dataframe
        df = pd.merge(df, player_median_mins, on='playerID', how='left')
        
        # Calculate minutes deviation from typical minutes
        df['mins_deviation'] = df['mins'] - df['median_mins']
        
        # Flag possible minutes restriction
        # Consider a player on minutes restriction if they played significantly less than their median
        # but still played in the game (e.g., more than 5 minutes)
        df['minutes_restriction'] = ((df['mins_deviation'] < -8) & (df['mins'] > 5)).astype(int)
        
        # Also flag games where minutes are returning to normal after a restriction
        # This helps identify players coming back from injury
        df['returning_from_restriction'] = (
            (df['mins_deviation'] > 5) & 
            (df.groupby('playerID')['minutes_restriction'].shift(1) == 1)
        ).astype(int)
        
        # Create a feature for recent minutes volatility
        # High volatility might indicate a player with changing role or returning from injury
        df['mins_volatility'] = df.groupby('playerID')['mins'].rolling(window=5, min_periods=2).std().reset_index(level=0, drop=True)
        df['mins_volatility'] = df['mins_volatility'].fillna(0)
        
        logging.info("Created minutes restriction features")
    
    # Detect mid-season team changes
    if 'teamAbv' in df.columns:
        # For each player, check if their team changes from one game to the next
        df['prev_team'] = df.groupby('playerID')['teamAbv'].shift(1)
        df['team_changed'] = (df['teamAbv'] != df['prev_team']).astype(int)
        
        # Count games since team change (useful for adaptation period)
        # Reset counter at each team change
        df['games_since_team_change'] = 0
        
        # Process each player separately
        for player_id, player_df in df.groupby('playerID'):
            if player_df['team_changed'].sum() == 0:
                continue  # No team changes for this player
                
            # Initialize counter
            counter = 0
            indices = player_df.index
            
            for i, idx in enumerate(indices):
                if i == 0:  # First game, no previous team
                    continue
                    
                if df.loc[idx, 'team_changed'] == 1:
                    # Reset counter at team change
                    counter = 1
                else:
                    # Increment counter
                    counter += 1
                
                df.loc[idx, 'games_since_team_change'] = counter
        
        # Create a feature that indicates the adaptation period after a trade
        # (first 10 games with new team are typically an adjustment period)
        df['trade_adaptation_period'] = ((df['team_changed'] == 1) | 
                                       ((df['games_since_team_change'] > 0) & 
                                        (df['games_since_team_change'] <= 10))).astype(int)
        
        logging.info("Created team change features")
    
    # Create features for back-to-back games which can indicate fatigue
    # This is already handled in the rest features, but we'll add it here for completeness
    if 'days_rest' in df.columns:
        # Update the impact of back-to-back games on different player age groups
        if 'Age' in df.columns:
            # Younger players (under 25) are less affected by back-to-backs
            young_mask = (df['Age'] < 25) & (df['days_rest'] < 2)
            df.loc[young_mask, 'b2b_age_impact'] = -0.1
            
            # Mid-career players (25-32) are moderately affected
            mid_mask = (df['Age'] >= 25) & (df['Age'] <= 32) & (df['days_rest'] < 2)
            df.loc[mid_mask, 'b2b_age_impact'] = -0.2
            
            # Older players (33+) are more affected by back-to-backs
            old_mask = (df['Age'] > 32) & (df['days_rest'] < 2)
            df.loc[old_mask, 'b2b_age_impact'] = -0.3
            
            # Default for non-back-to-back games
            df['b2b_age_impact'] = df['b2b_age_impact'].fillna(0)
            
            logging.info("Created age-specific back-to-back impact features")
    
    logging.info(f"Created player-specific factors with {len(df)} rows")
    return df

@time_function
@cache_result('team_lineup_features')
def create_team_lineup_features(game_data):
    """
    Create features based on team lineup combinations and their effectiveness
    
    Args:
        game_data (pd.DataFrame): DataFrame containing game data
        
    Returns:
        pd.DataFrame: DataFrame with additional team lineup features
    """
    # Make a copy of the input DataFrame
    df = game_data.copy()
    
    logging.info("Creating team lineup features...")
    
    # Create features for lineup continuity
    # Check if we have necessary columns
    has_starter_info = 'starter' in df.columns or 'gameStarter' in df.columns
    starter_col = 'starter' if 'starter' in df.columns else 'gameStarter' if 'gameStarter' in df.columns else None
    
    # Initialize features
    df['team_lineup_continuity'] = 0.0
    df['player_role_consistency'] = 0.0
    df['lineup_chemistry'] = 0.0
    
    # Check if we have necessary date information
    if 'game_date' not in df.columns:
        # Try to extract date from gameID
        if 'gameID' in df.columns:
            df['game_date'] = pd.to_datetime(
                df['gameID'].astype(str).str.split('_', expand=True)[0], 
                format='%Y%m%d', 
                errors='coerce'
            )
        else:
            logging.warning("Cannot create lineup features: missing date information")
            return df
    
    # Process team lineup continuity
    if has_starter_info and starter_col is not None:
        # Convert to int if needed
        if df[starter_col].dtype != 'int64':
            df['is_starter'] = df[starter_col].astype(int)
        else:
            df['is_starter'] = df[starter_col]
            
        # Process each team separately
        for team, team_df in df.groupby('teamAbv'):
            # Sort by date
            team_df = team_df.sort_values('game_date')
            
            # Process each date
            for date, date_df in team_df.groupby('game_date'):
                # Get past games for this team
                past_games = team_df[team_df['game_date'] < date]
                
                # Skip if not enough history
                if len(past_games) < 5:
                    continue
                
                # Get most recent games (last 10)
                recent_games = past_games.sort_values('game_date', ascending=False).head(10)
                
                # Get starters for each recent game
                if len(recent_games) > 0:
                    game_starters = recent_games[recent_games['is_starter'] == 1].groupby('game_date')['playerID'].apply(list)
                    
                    # Calculate lineup consistency
                    lineup_counts = {}
                    for starter_list in game_starters:
                        # Create hashable key from sorted players
                        key = tuple(sorted(starter_list))
                        lineup_counts[key] = lineup_counts.get(key, 0) + 1
                    
                    # Calculate continuity score - how often most common lineup is used
                    if lineup_counts:
                        max_count = max(lineup_counts.values())
                        continuity = max_count / len(game_starters)
                        
                        # Apply continuity score to all players in this game
                        current_indices = date_df.index
                        df.loc[current_indices, 'team_lineup_continuity'] = continuity
                
                # Calculate player role consistency
                for _, player_row in date_df.iterrows():
                    player_id = player_row['playerID']
                    
                    # Get past games for this player
                    player_past = recent_games[recent_games['playerID'] == player_id]
                    
                    # Need at least 3 games for consistency calculation
                    if len(player_past) >= 3 and 'mins' in player_past.columns:
                        # Calculate coefficient of variation for minutes played
                        mins_mean = player_past['mins'].mean()
                        mins_std = player_past['mins'].std()
                        
                        if mins_mean > 0:
                            cv = mins_std / mins_mean
                            # Convert to consistency score (1 is most consistent)
                            consistency = 1 / (1 + cv) if cv > 0 else 1.0
                            df.loc[player_row.name, 'player_role_consistency'] = consistency
                    
                    # Calculate lineup chemistry - players who frequently play together
                    if len(player_past) >= 3:
                        # For each game this player was in, which other players were also in the game
                        player_game_dates = player_past['game_date'].unique()
                        
                        # If we have date info for this player's games
                        if len(player_game_dates) > 0:
                            # Get all teammates in these games
                            teammates = recent_games[
                                (recent_games['game_date'].isin(player_game_dates)) & 
                                (recent_games['playerID'] != player_id)
                            ]
                            
                            # Find current teammates in this game
                            current_teammates = date_df[date_df['playerID'] != player_id]['playerID'].unique()
                            
                            # Calculate how many current teammates are familiar to this player
                            if len(current_teammates) > 0:
                                familiar_teammates = teammates['playerID'].unique()
                                overlap_count = sum(1 for t in current_teammates if t in familiar_teammates)
                                chemistry = overlap_count / len(current_teammates)
                                df.loc[player_row.name, 'lineup_chemistry'] = chemistry
    
    # Calculate team cohesion based on player tenure with team
    if 'player_tenure' not in df.columns and 'teamAbv' in df.columns and 'playerID' in df.columns:
        # For each player, calculate how many games they've played with their current team
        player_team_counts = {}
        for team, team_df in df.groupby(['teamAbv']):
            for player, player_df in team_df.groupby(['playerID']):
                # Sort by date
                player_df = player_df.sort_values('game_date')
                
                # For each game, calculate games with team so far
                for idx, row in player_df.iterrows():
                    key = (team, player)
                    # Get current count or initialize to 0
                    current_count = player_team_counts.get(key, 0)
                    # Store count for this game
                    df.loc[idx, 'player_team_games'] = current_count
                    # Increment count for next game
                    player_team_counts[key] = current_count + 1
        
        # Convert games to scaled tenure metric (0-1)
        if 'player_team_games' in df.columns:
            max_games = df['player_team_games'].max()
            if max_games > 0:
                df['player_tenure'] = df['player_team_games'] / max_games
                # Apply diminishing returns curve: 1 - exp(-games/30)
                df['player_tenure'] = 1 - np.exp(-df['player_team_games'] / 30)
                # Drop intermediate column
                df = df.drop(columns=['player_team_games'])
    
    # Fill any missing values with neutral values
    lineup_cols = ['team_lineup_continuity', 'player_role_consistency', 
                  'lineup_chemistry', 'player_tenure']
    for col in lineup_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)
    
    logging.info(f"Created team lineup features with {len(df)} rows")
    return df

def create_feature_matrix(processed_data, features_to_use=None, target_cols=None):
    """
    Create a feature matrix for modeling
    
    Args:
        processed_data (pd.DataFrame): DataFrame containing processed game data
        features_to_use (list, optional): List of feature columns to include
        target_cols (list, optional): List of target columns for prediction
        
    Returns:
        tuple: X (feature matrix), y (target matrix), feature_names (list of feature names)
    """
    try:
        # Make a copy of the input DataFrame
        df = processed_data.copy()
        
        # Default target columns if not specified
        if target_cols is None:
            # Try common variations of column names
            target_candidates = [
                ['pts', 'reb', 'ast', 'stl', 'blk', 'TOV_x', 'plusMinus'],  # Original names
                ['PTS', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'plusMinus'],    # Uppercase variations
                ['points', 'rebounds', 'assists', 'steals', 'blocks', 'turnovers', 'plus_minus']  # Full names
            ]
            
            # Find the first set that has the most matches
            best_match = []
            for candidates in target_candidates:
                matches = [col for col in candidates if col in df.columns]
                if len(matches) > len(best_match):
                    best_match = matches
            
            target_cols = best_match
            
            # If we still have no matches, look for any statistical columns
            if not target_cols:
                logging.warning("No standard target columns found. Looking for statistical columns...")
                # Look for any columns that might be stats
                stat_pattern = re.compile(r'(pts|points|reb|rebounds|ast|assists|stl|steals|blk|blocks|tov|turnovers)', re.IGNORECASE)
                target_cols = [col for col in df.columns if stat_pattern.search(col)]
                logging.info(f"Found potential target columns: {target_cols}")
            
            if not target_cols:
                logging.error("No viable target columns found in the data")
                # Return empty dataframes
                return pd.DataFrame(), pd.DataFrame(), []
        
        # Default feature columns if not specified
        if features_to_use is None:
            # Player profile features
            profile_cols = ['MP', 'Age', 'home_game', 'days_rest', 'USG%', 'WS/48', 'OBPM', 'DBPM', 'BPM', 'opp_strength']
            
            # Season average features
            season_avg_cols = ['PTS', 'TRB', 'AST', 'STL', 'BLK', 'TOV_y', 'FG%', '3P%', 'FT%']
            
            # Advanced metrics
            advanced_cols = ['eFG%', 'TS%', 'AST_TO_ratio', 'PPS']
            
            # Recent performance features
            recent_cols = [col for col in df.columns if (
                col.startswith('last3_') or 
                col.startswith('last10_')
            )]
            
            # Trend features
            trend_cols = [col for col in df.columns if (
                col.endswith('_trend') or 
                col.endswith('_w_trend') or
                col.endswith('_w_avg')
            )]
            
            # Consistency features
            consistency_cols = [col for col in df.columns if col.endswith('_consistency')]
            
            # Player-specific factors
            player_specific_cols = [
                'minutes_restriction', 'returning_from_restriction', 
                'mins_volatility', 'team_changed', 'games_since_team_change',
                'trade_adaptation_period', 'b2b_age_impact'
            ]
            
            # Season-to-date features
            std_cols = [col for col in df.columns if col.startswith('std_')]
            
            # Combine all potential feature columns
            all_feature_cols = (
                profile_cols + season_avg_cols + advanced_cols + recent_cols + 
                trend_cols + consistency_cols + player_specific_cols + std_cols
            )
            
            # Check which columns actually exist in the dataframe
            features_to_use = [col for col in all_feature_cols if col in df.columns]
            
            # If no features were found, fall back to using all available columns except targets
            if not features_to_use:
                logging.warning("No predefined features found. Using all available columns except targets.")
                features_to_use = [col for col in df.columns if col not in target_cols]
        
        # Drop rows with missing target values
        df = df.dropna(subset=target_cols)
        
        # Prepare the feature matrix and target
        X = df[features_to_use].copy()
        y = df[target_cols].copy()
        
        # Handle missing values in features more robustly
        # First check if we have any data
        if X.shape[0] == 0:
            logging.error("No data available for creating feature matrix")
            return X, y, features_to_use
        
        # For numeric columns, fill with mean
        numeric_cols = X.select_dtypes(include=['int', 'float']).columns
        for col in numeric_cols:
            # Check if column has any NA values to avoid ambiguous Series truth value
            na_mask = X[col].isna()
            if na_mask.any():
                # Check if all values are NA
                if na_mask.all():
                    X[col] = X[col].fillna(0)
                else:
                    # Fill with mean
                    X[col] = X[col].fillna(X[col].mean())
    
        # For other columns, fill with mode or a default value
        non_numeric_cols = X.select_dtypes(exclude=['int', 'float']).columns
        for col in non_numeric_cols:
            # Check if column has any NA values to avoid ambiguous Series truth value
            na_mask = X[col].isna()
            if na_mask.any():
                # Check if all values are NA
                if na_mask.all():
                    X[col] = X[col].fillna("unknown")
                else:
                    # Fill with mode if available
                    mode_values = X[col].mode()
                    if len(mode_values) > 0:
                        X[col] = X[col].fillna(mode_values[0])
                    else:
                        X[col] = X[col].fillna("unknown")
    
        # Check for and handle any remaining NaN values
        if X.isnull().any().any():
            logging.warning(f"Still have NaN values after filling. Dropping affected rows.")
            valid_rows = ~X.isnull().any(axis=1)
            X = X.loc[valid_rows]
            y = y.loc[valid_rows]
        
        # Get feature names
        feature_names = X.columns.tolist()
        
        logging.info(f"Created feature matrix with {X.shape[0]} rows and {X.shape[1]} columns")
        return X, y, feature_names
        
    except Exception as e:
        logging.error(f"Error in create_feature_matrix: {str(e)}")
        # Return empty dataframes to avoid crashing the pipeline
        empty_X = pd.DataFrame()
        empty_y = pd.DataFrame(columns=target_cols if target_cols else [])
        return empty_X, empty_y, []

@time_function
def detect_and_handle_redundant_features(df, importance_threshold=0.01, correlation_threshold=0.9, verbose=True):
    """
    Detect and handle redundant features that may confuse the model
    
    Args:
        df (pd.DataFrame): DataFrame with features
        importance_threshold (float): Threshold for considering a feature important
        correlation_threshold (float): Threshold for considering features highly correlated
        verbose (bool): Whether to print detailed information
        
    Returns:
        pd.DataFrame: DataFrame with redundant features removed
    """
    # Make a copy of the input dataframe
    df_copy = df.copy()
    
    # Get current feature importance data
    importance = load_feature_importance()
    
    if not importance:
        logging.warning("No feature importance data available. Cannot detect redundant features based on importance.")
        return df_copy
    
    # Identify potential redundant features based on naming patterns
    redundant_patterns = [
        # Per-36 vs raw stats
        ('pts', 'pts_per36'),
        ('reb', 'reb_per36'),
        ('ast', 'ast_per36'),
        ('stl', 'stl_per36'),
        ('blk', 'blk_per36'),
        ('TOV_x', 'TOV_x_per36'),
        # Regular vs weighted averages
        ('last3_pts_avg', 'last3_pts_w_avg'),
        ('last10_pts_avg', 'last10_pts_w_avg'),
        ('last3_reb_avg', 'last3_reb_w_avg'),
        ('last10_reb_avg', 'last10_reb_w_avg'),
        ('last3_ast_avg', 'last3_ast_w_avg'),
        ('last10_ast_avg', 'last10_ast_w_avg'),
        # Regular vs weighted trends
        ('pts_trend', 'pts_w_trend'),
        ('reb_trend', 'reb_w_trend'),
        ('ast_trend', 'ast_w_trend'),
        # Similar metrics
        ('eFG%', 'TS%'),
        ('PTS', 'pts'),
        ('TRB', 'reb'),
        ('AST', 'ast'),
        ('USG%', 'USG'),
    ]
    
    # Track columns to drop
    to_drop = []
    drop_reasons = {}
    
    # Check redundancy within identified pairs
    for col1, col2 in redundant_patterns:
        # Skip if either column doesn't exist
        if col1 not in df_copy.columns or col2 not in df_copy.columns:
            continue
        
        # Get importance scores (default to 0 if not found)
        imp1 = importance.get(col1, 0)
        imp2 = importance.get(col2, 0)
        
        # If both are below threshold, drop the second one
        if imp1 < importance_threshold and imp2 < importance_threshold:
            to_drop.append(col2)
            drop_reasons[col2] = f"Low importance pair with {col1} (imp1={imp1:.4f}, imp2={imp2:.4f})"
            continue
        
        # Calculate correlation if both columns have numeric data
        if df_copy[col1].dtype.kind in 'if' and df_copy[col2].dtype.kind in 'if':
            try:
                correlation = df_copy[col1].corr(df_copy[col2])
                # If correlation is NaN, skip this pair
                if pd.isna(correlation):
                    continue
                    
                # If highly correlated, keep the one with higher importance
                if abs(correlation) > correlation_threshold:
                    if imp1 >= imp2:
                        to_drop.append(col2)
                        drop_reasons[col2] = f"Correlated with {col1} (corr={correlation:.2f}, imp1={imp1:.4f}, imp2={imp2:.4f})"
                    else:
                        to_drop.append(col1)
                        drop_reasons[col1] = f"Correlated with {col2} (corr={correlation:.2f}, imp1={imp1:.4f}, imp2={imp2:.4f})"
            except Exception as e:
                logging.warning(f"Error calculating correlation between {col1} and {col2}: {str(e)}")
    
    # Find highly correlated features more generally
    numeric_df = df_copy.select_dtypes(include=['float64', 'float32', 'int64', 'int32'])
    
    # Skip the correlation calculation if the dataframe is too big
    if numeric_df.shape[1] < 100:  # Only run on dataframes with reasonable number of columns
        try:
            # Calculate correlation matrix
            corr_matrix = numeric_df.corr()
            
            # Get upper triangle of correlation matrix
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            
            # Find features with correlation greater than threshold
            high_corr_pairs = []
            for col in upper.columns:
                # Get highly correlated features for this column
                correlated_features = upper.index[upper[col].abs() > correlation_threshold].tolist()
                
                for corr_feat in correlated_features:
                    if col != corr_feat:  # Avoid self-correlation (though not needed with upper triangle)
                        high_corr_pairs.append((col, corr_feat, upper.loc[corr_feat, col]))
            
            # For each pair, decide which to drop based on feature importance
            for col1, col2, corr in high_corr_pairs:
                # Skip if either is already in to_drop
                if col1 in to_drop or col2 in to_drop:
                    continue
                    
                imp1 = importance.get(col1, 0)
                imp2 = importance.get(col2, 0)
                
                # Drop the one with lower importance
                if imp1 < imp2:
                    to_drop.append(col1)
                    drop_reasons[col1] = f"Correlated with {col2} (corr={corr:.2f}, imp1={imp1:.4f}, imp2={imp2:.4f})"
                else:
                    to_drop.append(col2)
                    drop_reasons[col2] = f"Correlated with {col1} (corr={corr:.2f}, imp1={imp1:.4f}, imp2={imp2:.4f})"
        except Exception as e:
            logging.warning(f"Error detecting general feature correlations: {str(e)}")
    
    # Remove duplicates from to_drop
    to_drop = list(set(to_drop))
    
    # Log the dropped columns and reasons
    if to_drop and verbose:
        logging.info(f"Removing {len(to_drop)} redundant features:")
        for col in to_drop:
            logging.info(f"  - {col}: {drop_reasons.get(col, 'Unknown reason')}")
    
    # Drop the columns
    df_copy = df_copy.drop(columns=to_drop, errors='ignore')
    
    return df_copy


def detect_unused_features(df, importance_threshold=0.005, min_pct_drop=0.1, verbose=True):
    """
    Detect and remove features that show low importance in the model
    
    Args:
        df (pd.DataFrame): DataFrame with features
        importance_threshold (float): Threshold for considering a feature important
        min_pct_drop (float): Minimum percentage of columns to consider dropping
        verbose (bool): Whether to print detailed information
        
    Returns:
        pd.DataFrame: DataFrame with unused features removed, or original if no importance data
    """
    # Make a copy of the input dataframe
    df_copy = df.copy()
    
    # Get current feature importance data
    importance = load_feature_importance()
    
    if not importance:
        logging.warning("No feature importance data available. Cannot detect unused features.")
        return df_copy
    
    # Always keep certain columns regardless of importance
    keep_columns = [
        'longName', 'opponent', 'playerID', 'teamAbv', 'game_date',
        'pts', 'reb', 'ast', 'stl', 'blk', 'TOV_x', 'plusMinus'  # Target columns
    ]
    
    # Identify features with importance below threshold
    low_importance = {}
    for col in df_copy.columns:
        # Skip columns we always want to keep
        if col in keep_columns:
            continue
            
        # Get importance score (default to 0 if not found)
        imp = importance.get(col, 0)
        
        # If below threshold, mark for potential removal
        if imp < importance_threshold:
            low_importance[col] = imp
    
    # Sort by importance (ascending)
    low_importance = {k: v for k, v in sorted(low_importance.items(), key=lambda item: item[1])}
    
    # Determine how many columns to drop (at least min_pct_drop of total)
    min_drop_count = max(int(len(df_copy.columns) * min_pct_drop), 1)
    drop_count = min(len(low_importance), min_drop_count)
    
    # Select columns to drop (lowest importance first)
    to_drop = list(low_importance.keys())[:drop_count]
    
    # Log the dropped columns
    if to_drop and verbose:
        logging.info(f"Removing {len(to_drop)} unused features with importance below {importance_threshold}:")
        for col in to_drop:
            logging.info(f"  - {col}: importance={low_importance.get(col, 0):.6f}")
    
    # Drop the columns
    df_copy = df_copy.drop(columns=to_drop, errors='ignore')
    
    return df_copy


def validate_derived_features(df, verbose=True):
    """
    Validate and fix issues in derived statistical features
    
    Args:
        df (pd.DataFrame): DataFrame with features
        verbose (bool): Whether to print detailed information
        
    Returns:
        pd.DataFrame: DataFrame with validated derived features
    """
    # Make a copy of the input dataframe
    df_copy = df.copy()
    
    # Define validation checks for different statistical categories
    validations = {
        # Per-36 stats should be within reasonable ranges
        'per36': {
            'pattern': r'_per36$',
            'cols': [],
            'checks': [
                ('pts_per36', 0, 60),  # Max ~60 pts per 36 min is reasonable
                ('reb_per36', 0, 40),  # Max ~40 reb per 36 min is reasonable
                ('ast_per36', 0, 30),  # Max ~30 ast per 36 min is reasonable
                ('stl_per36', 0, 15),  # Max ~15 stl per 36 min is reasonable
                ('blk_per36', 0, 15),  # Max ~15 blk per 36 min is reasonable
                ('TOV_x_per36', 0, 20)  # Max ~20 tov per 36 min is reasonable
            ]
        },
        # Efficiency metrics
        'efficiency': {
            'pattern': r'(eFG%|TS%|PPS|AST_TO_ratio)',
            'cols': ['eFG%', 'TS%', 'PPS', 'AST_TO_ratio'],
            'checks': [
                ('eFG%', 0, 1),  # Effective field goal % should be 0-1
                ('TS%', 0, 1),   # True shooting % should be 0-1
                ('PPS', 0, 3),   # Points per shot typically 0-3
                ('AST_TO_ratio', 0, 20)  # Assist to turnover ratio rarely above 20
            ]
        },
        # Trend features
        'trend': {
            'pattern': r'_trend$',
            'cols': [],
            'checks': []  # Will be dynamically populated
        },
        # Weighted average features
        'w_avg': {
            'pattern': r'_w_avg$',
            'cols': [],
            'checks': []  # Will be dynamically populated
        },
        # Consistency features
        'consistency': {
            'pattern': r'_consistency$',
            'cols': [],
            'checks': [
                ('overall_consistency', 0, 1),  # Consistency score should be 0-1
                ('pts_consistency', 0, 1),      # Consistency score should be 0-1
                ('reb_consistency', 0, 1),      # Consistency score should be 0-1
                ('ast_consistency', 0, 1)       # Consistency score should be 0-1
            ]
        }
    }
    
    # Find all columns matching patterns
    for category, config in validations.items():
        if not config['cols']:  # If not explicitly specified
            pattern = re.compile(config['pattern'])
            config['cols'] = [col for col in df_copy.columns if pattern.search(col)]
    
    # Add dynamic checks for trend features
    trend_cols = validations['trend']['cols']
    for col in trend_cols:
        # Base stat from trend column name
        base_stat = col.replace('_trend', '')
        base_stat = base_stat.replace('_w', '')
        
        # Skip non-numeric columns
        if base_stat not in df_copy.columns or df_copy[base_stat].dtype.kind not in 'if':
            continue
        
        # Calculate reasonable min/max based on the standard deviation of the base stat
        std_val = df_copy[base_stat].std()
        if pd.notna(std_val) and std_val > 0:
            min_val = -5 * std_val  # 5 standard deviations below
            max_val = 5 * std_val   # 5 standard deviations above
            validations['trend']['checks'].append((col, min_val, max_val))
    
    # Add dynamic checks for weighted average features
    w_avg_cols = validations['w_avg']['cols']
    for col in w_avg_cols:
        # Base stat from weighted average column name
        base_stat = col.replace('_w_avg', '')
        if 'last3_' in base_stat:
            base_stat = base_stat.replace('last3_', '')
        elif 'last10_' in base_stat:
            base_stat = base_stat.replace('last10_', '')
        
        # Skip non-numeric columns
        if base_stat not in df_copy.columns or df_copy[base_stat].dtype.kind not in 'if':
            continue
        
        # Calculate reasonable min/max based on the min/max of the base stat
        min_val = df_copy[base_stat].min() if pd.notna(df_copy[base_stat].min()) else 0
        max_val = df_copy[base_stat].max() if pd.notna(df_copy[base_stat].max()) else 100
        validations['w_avg']['checks'].append((col, min_val, max_val))
    
    # Track validation issues
    validation_issues = {}
    
    # Apply all validation checks
    for category, config in validations.items():
        for col, min_val, max_val in config['checks']:
            if col in df_copy.columns:
                # Count out-of-range values
                below_min = (df_copy[col] < min_val).sum()
                above_max = (df_copy[col] > max_val).sum()
                
                # Log and fix issues
                if below_min > 0 or above_max > 0:
                    validation_issues[col] = {'below_min': below_min, 'above_max': above_max}
                    
                    # Clip values to valid range
                    df_copy[col] = df_copy[col].clip(lower=min_val, upper=max_val)
    
    # Log validation results
    if validation_issues and verbose:
        logging.info(f"Validation detected issues in {len(validation_issues)} derived features:")
        for col, issues in validation_issues.items():
            logging.info(f"  - {col}: {issues['below_min']} values below min, {issues['above_max']} values above max (fixed)")
    
    return df_copy


def check_cache_freshness(cache_file, input_data=None, recache_threshold_days=3):
    """
    Check if a cache file should be considered stale and regenerated
    
    Args:
        cache_file (str): Path to the cache file
        input_data (pd.DataFrame, optional): Input data for comparison
        recache_threshold_days (int): Number of days after which cache is considered stale
        
    Returns:
        bool: True if cache should be regenerated, False otherwise
    """
    # If file doesn't exist, it needs to be generated
    if not os.path.exists(cache_file):
        return True
    
    # Check age of cache file
    mod_time = os.path.getmtime(cache_file)
    age_days = (time.time() - mod_time) / (60 * 60 * 24)
    
    # If older than threshold, regenerate
    if age_days > recache_threshold_days:
        logging.info(f"Cache file {os.path.basename(cache_file)} is {age_days:.1f} days old (threshold: {recache_threshold_days}). Will regenerate.")
        return True
    
    # If input data is provided, check if row count matches
    if input_data is not None:
        try:
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
                
                # Check if row counts match
                if len(cached_data) != len(input_data):
                    logging.info(f"Cache file {os.path.basename(cache_file)} has {len(cached_data)} rows, but input data has {len(input_data)} rows. Will regenerate.")
                    return True
                    
                # Check if column counts match
                if len(cached_data.columns) < len(input_data.columns) * 0.8:  # Allow for some column differences
                    logging.info(f"Cache file {os.path.basename(cache_file)} has significantly different columns. Will regenerate.")
                    return True
        except Exception as e:
            logging.warning(f"Error checking cache file contents: {str(e)}. Will regenerate.")
            return True
    
    # Cache is fresh
    return False


def engineer_all_features(processed_data, team_ratings_path=None, selective=True, use_cache=True, 
                         cache_key=None, save_to_cache=True, is_prediction=False, remove_unnecessary_columns=True,
                         use_weighted_averages=True, remove_redundant_features=True, recache_threshold_days=3,
                         validate_derived_values=True):
    """
    Apply all feature engineering steps to create a complete set of features
    
    Args:
        processed_data (pd.DataFrame): DataFrame containing processed game data
        team_ratings_path (str, optional): Path to the team ratings file
        selective (bool): Whether to selectively create only important features
        use_cache (bool): Whether to check for cached results before computing
        cache_key (str, optional): Custom cache key, if None will use current date
        save_to_cache (bool): Whether to save results to cache
        is_prediction (bool): Whether this is for prediction (affects caching behavior)
        remove_unnecessary_columns (bool): Whether to remove non-feature columns
        use_weighted_averages (bool): Whether to use weighted averages for trends
        remove_redundant_features (bool): Whether to detect and remove redundant features
        recache_threshold_days (int): Number of days after which cache is considered stale
        validate_derived_values (bool): Whether to validate and fix derived values
        
    Returns:
        pd.DataFrame: DataFrame with all engineered features
    """
    logging.info("Starting feature engineering process...")
    
    # For prediction mode, disable caching
    if is_prediction:
        use_cache = False
        save_to_cache = False
    
    # Generate a cache key based on data characteristics if not provided
    if cache_key is None:
        # Use current date + row count as a simple cache key
        date_str = datetime.now().strftime("%Y%m%d")
        row_count = len(processed_data)
        cache_key = f"all_features_{date_str}_{row_count}"
    
    # Check if we have a cached version
    if use_cache:
        cache_file = os.path.join(CACHE_DIR, f"{cache_key}.pkl")
        
        # Check if the cache is fresh enough or needs regeneration
        if check_cache_freshness(cache_file, processed_data, recache_threshold_days):
            logging.info("Cache considered stale, will regenerate features")
        else:
            try:
                with open(cache_file, 'rb') as f:
                    df = pickle.load(f)
                    logging.info(f"Loaded cached engineered features from {cache_file}")
                    return df
            except Exception as e:
                logging.warning(f"Error loading cached features: {str(e)}")
    
    # Define helper functions to unwrap the decorated functions
    def create_matchup_unwrapped(data):
        return create_matchup_features(data, _use_cache=not is_prediction)
        
    def create_rest_unwrapped(data):
        return create_rest_features(data, _use_cache=not is_prediction)
        
    def create_trend_unwrapped(data, selective=True, use_weighted_avgs=True):
        return create_trend_features(data, selective=selective, use_weighted_averages=use_weighted_avgs, _use_cache=not is_prediction)
        
    def create_consistency_unwrapped(data, selective=True):
        return create_player_consistency_features(data, selective=selective, _use_cache=not is_prediction)
    
    def create_defensive_matchup_unwrapped(data, team_ratings_path=None):
        return create_defensive_matchup_features(data, team_ratings_path=team_ratings_path, _use_cache=not is_prediction)
    
    def create_time_weighted_unwrapped(data, decay_factor=0.9, max_games=20):
        return create_time_weighted_features(data, decay_factor=decay_factor, max_games=max_games, _use_cache=not is_prediction)
    
    def create_team_lineup_unwrapped(data):
        return create_team_lineup_features(data, _use_cache=not is_prediction)
        
    # Apply data quality checks before feature engineering if available
    if QUALITY_CHECKS_AVAILABLE:
        logging.info("Applying data quality checks before feature engineering")
        processed_data = fix_invalid_values(processed_data)
        processed_data = resolve_turnover_columns(processed_data)
    
    # Apply each feature engineering step in a logical order
    
    # Step 1: Basic matchup and game information features
    df = create_matchup_unwrapped(processed_data)
    
    # Step 2: Rest and schedule-related features
    df = create_rest_unwrapped(df)
    
    # Step 3: Add advanced offensive metrics
    df = create_advanced_offensive_features(df)
    
    # Step 4: Create opponent strength features
    df = create_opp_strength_features(df, team_ratings_path)
    
    # Step 5: Player-specific factors (minutes restrictions, team changes)
    df = create_player_specific_factors(df)
    
    # Step 6: Create trend features (including weighted averages if requested)
    df = create_trend_unwrapped(df, selective=selective, use_weighted_avgs=use_weighted_averages)
    
    # Step 7: Player consistency features
    df = create_consistency_unwrapped(df, selective=selective)
    
    # Step 7.1: Defensive matchup features
    team_ratings_path = os.path.join(DATA_DIR, "standings", f"team_ratings_{datetime.now().strftime('%Y%m%d')}.csv")
    if not os.path.exists(team_ratings_path):
        # Try with yesterday's date if today's not available
        yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y%m%d')
        team_ratings_path = os.path.join(DATA_DIR, "standings", f"team_ratings_{yesterday}.csv")
    
    df = create_defensive_matchup_unwrapped(df, team_ratings_path=team_ratings_path)
    
    # Step 7.2: Time-weighted features with recency bias
    df = create_time_weighted_unwrapped(df)
    
    # Step 7.3: Team lineup context features
    df = create_team_lineup_unwrapped(df)
    
    # Step 8: Usage features
    df = create_usage_features(df)
    
    # Step 9: Validate derived features if requested
    if validate_derived_values:
        df = validate_derived_features(df)
    
    # Step 10: Remove redundant and unused features if requested
    if remove_redundant_features:
        df = detect_and_handle_redundant_features(df)
        df = detect_unused_features(df)
    
    # Remove unnecessary columns if requested
    if remove_unnecessary_columns:
        # Define columns to keep
        keep_cols = [
            # Feature columns we want to keep
            'MP', 'Age', 'home_game', 'days_rest', 'USG%', 'WS/48', 'OBPM', 'DBPM', 'BPM', 'opp_strength',
            'PTS', 'TRB', 'AST', 'STL', 'BLK', 'TOV_y', 'FG%', '3P%', 'FT%',
            
            # Advanced metrics
            'eFG%', 'TS%', 'AST_TO_ratio', 'PPS',
            
            # Player-specific factors
            'minutes_restriction', 'returning_from_restriction', 'mins_volatility',
            'team_changed', 'games_since_team_change', 'trade_adaptation_period', 'b2b_age_impact'
        ]
        
        # Add other pattern-based feature columns
        pattern_cols = [
            col for col in df.columns if (
                col.startswith('last3_') or 
                col.startswith('last10_') or 
                col.endswith('_trend') or 
                col.endswith('_w_avg') or
                col.endswith('_w_trend') or
                col.endswith('_consistency') or
                col.startswith('std_')
            )
        ]
        
        # Create the final list of columns to keep
        all_keep_cols = keep_cols + pattern_cols
        
        # Keep only essential identification columns
        # We'll only keep longName and opponent for predictions,
        # but exclude playerID, teamID, and other unnecessary IDs
        if 'longName' in df.columns:
            all_keep_cols.append('longName')
        if 'opponent' in df.columns:
            all_keep_cols.append('opponent')
        
        # Keep target columns for training
        target_cols = ['pts', 'reb', 'ast', 'stl', 'blk', 'TOV_x', 'plusMinus']
        for col in target_cols:
            if col in df.columns:
                all_keep_cols.append(col)
        
        # Get unique column list
        all_keep_cols = list(set(all_keep_cols))
        
        # Check which columns exist in our dataframe
        existing_cols = [col for col in all_keep_cols if col in df.columns]
        
        # Create a copy with only the columns we want to keep
        columns_to_drop = [col for col in df.columns if col not in existing_cols]
        if columns_to_drop:
            logging.info(f"Removing {len(columns_to_drop)} unnecessary columns from engineered data")
            df = df[existing_cols].copy()
    
    # Save to cache if requested
    if save_to_cache:
        try:
            cache_file = os.path.join(CACHE_DIR, f"{cache_key}.pkl")
            with open(cache_file, 'wb') as f:
                pickle.dump(df, f)
                logging.info(f"Saved engineered features to cache: {cache_file}")
        except Exception as e:
            logging.warning(f"Error saving to cache: {str(e)}")
    
    logging.info(f"Completed feature engineering with {df.shape[0]} rows and {df.shape[1]} columns")
    return df

if __name__ == "__main__":
    import os
    import argparse
    from datetime import datetime
    
    # Define command line arguments for flexible feature engineering
    parser = argparse.ArgumentParser(description="Generate engineered features for NBA player performance prediction")
    
    # Data input/output options
    input_group = parser.add_argument_group('Data Options')
    input_group.add_argument("--input-file", type=str, help="Path to processed data CSV file (optional)")
    input_group.add_argument("--output-file", type=str, help="Path to save engineered data CSV file (optional)")
    input_group.add_argument("--team-ratings", type=str, help="Path to team ratings CSV file (optional)")
    
    # Feature generation options
    feature_group = parser.add_argument_group('Feature Options')
    feature_group.add_argument("--all-features", action="store_true", 
                             help="Generate all features (not selective)")
    feature_group.add_argument("--feature-importance-threshold", type=float, default=0.01, 
                             help="Minimum importance threshold for including features (default: 0.01)")
    feature_group.add_argument("--update-importance", action="store_true", 
                             help="Update feature importance after model training")
    feature_group.add_argument("--model-file", type=str, 
                             help="Path to model file for updating feature importance")
    feature_group.add_argument("--disable-weighted-avgs", action="store_true", 
                             help="Disable weighted averages for trend features")
    feature_group.add_argument("--advanced-metrics", action="store_true", 
                             help="Calculate advanced metrics (TS%, eFG%, AST/TO) even if not selective")
    
    # Data quality options
    quality_group = parser.add_argument_group('Data Quality Options')
    quality_group.add_argument("--run-full-quality-checks", action="store_true", 
                             help="Run comprehensive data quality checks before feature engineering")
    quality_group.add_argument("--min-minutes", type=int, default=10, 
                             help="Minimum minutes threshold for low-minute players (default: 10)")
    
    # Caching options
    cache_group = parser.add_argument_group('Caching Options')
    cache_group.add_argument("--no-cache", action="store_true", 
                           help="Don't use cached results")
    cache_group.add_argument("--cache-key", type=str, 
                           help="Custom cache key for storing/retrieving cached results")
    cache_group.add_argument("--cleanup-cache", action="store_true", 
                           help="Clean up old cache files")
    cache_group.add_argument("--cache-max-age", type=int, default=7, 
                           help="Maximum age in days to keep cache files (default: 7)")
    cache_group.add_argument("--cache-dry-run", action="store_true", 
                           help="Show what cache files would be deleted without deleting")
    
    # Performance options
    perf_group = parser.add_argument_group('Performance Options')
    perf_group.add_argument("--profile", action="store_true", 
                          help="Run with detailed timing for profiling feature creation")
    
    args = parser.parse_args()
    
    # Ensure directories exist
    ensure_directories_exist()
    
    # Check if we should just clean up the cache
    if args.cleanup_cache and not args.input_file:
        # Just run cache cleanup
        deleted_count = cleanup_cache(max_age_days=args.cache_max_age, dry_run=args.cache_dry_run)
        if args.cache_dry_run:
            logging.info(f"Would delete {deleted_count} cache files")
        else:
            logging.info(f"Deleted {deleted_count} cache files")
        exit(0)
    
    # Check if we should just update feature importance
    if args.update_importance and args.model_file and not args.input_file:
        # Just update feature importance from model
        updated = update_feature_importance_from_model(
            model_file=args.model_file,
            threshold=args.feature_importance_threshold
        )
        
        if updated:
            num_features = len(updated)
            logging.info(f"Updated feature importance with {num_features} features")
            logging.info(f"Top 10 most important features:")
            top_features = sorted(updated.items(), key=lambda x: x[1], reverse=True)[:10]
            for feature, importance in top_features:
                logging.info(f"  {feature}: {importance:.4f}")
        else:
            logging.error("Failed to update feature importance")
        exit(0)
    
    # Get the latest processed data file if not specified
    if args.input_file:
        processed_data_path = args.input_file
    else:
        # Try to load from config, otherwise use default path
        try:
            from config import get_processed_data_path
            processed_data_path = get_processed_data_path()
        except ImportError:
            # Default fallback
            current_date = datetime.now().strftime("%Y%m%d")
            processed_data_path = f"/Users/lukesmac/Projects/nbaModel/data/processed/processed_nba_data_{current_date}.csv"
        
        # If today's file doesn't exist, try to find the latest one
        if not os.path.exists(processed_data_path):
            processed_dir = os.path.dirname(processed_data_path)
            if os.path.exists(processed_dir):
                processed_files = sorted(
                    [f for f in os.listdir(processed_dir) if f.startswith("processed_nba_data_")],
                    reverse=True
                )
                if processed_files:
                    processed_data_path = os.path.join(processed_dir, processed_files[0])
                else:
                    logging.error("No processed data file found")
                    exit(1)
    
    # Clean up cache if requested (before loading data)
    if args.cleanup_cache:
        deleted_count = cleanup_cache(max_age_days=args.cache_max_age, dry_run=args.cache_dry_run)
        if args.cache_dry_run:
            logging.info(f"Would delete {deleted_count} cache files")
        else:
            logging.info(f"Deleted {deleted_count} cache files")
    
    # Load the processed data
    try:
        logging.info(f"Loading processed data from: {processed_data_path}")
        processed_data = pd.read_csv(processed_data_path)
        logging.info(f"Loaded processed data with {processed_data.shape[0]} rows and {processed_data.shape[1]} columns")
    except Exception as e:
        logging.error(f"Error loading processed data: {str(e)}")
        exit(1)
    
    # Apply data quality checks if requested and available
    if args.run_full_quality_checks and QUALITY_CHECKS_AVAILABLE:
        from src.data_quality import run_all_quality_checks
        logging.info("Running full data quality checks before feature engineering")
        processed_data = run_all_quality_checks(processed_data, min_minutes=args.min_minutes)
        logging.info(f"Data quality checks completed, DataFrame now has {processed_data.shape[1]} columns")
    
    # Apply feature engineering
    engineered_data = engineer_all_features(
        processed_data,
        team_ratings_path=args.team_ratings,
        selective=not args.all_features,
        use_cache=not args.no_cache,
        cache_key=args.cache_key,
        save_to_cache=True,
        use_weighted_averages=not args.disable_weighted_avgs
    )
    
    # Determine output path
    if args.output_file:
        output_path = args.output_file
    else:
        # Try to load from config, otherwise use default path
        try:
            from config import get_engineered_data_path
            output_path = get_engineered_data_path()
        except ImportError:
            # Default fallback
            current_date = datetime.now().strftime("%Y%m%d")
            output_dir = "/Users/lukesmac/Projects/nbaModel/data/engineered"
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"engineered_nba_data_{current_date}.csv")
    
    # Save engineered data
    engineered_data.to_csv(output_path, index=False)
    logging.info(f"Saved engineered data to {output_path}")
    
    # Print summary of features generated
    feature_count = engineered_data.shape[1] - processed_data.shape[1]
    logging.info(f"Generated {feature_count} new features")
    
    # If profiling was requested, print detailed timing information
    if args.profile:
        # Sort functions by execution time (if available in globals)
        timing_info = {}
        for func_name in [
            'create_matchup_features', 'create_rest_features', 'create_trend_features', 
            'create_opp_strength_features', 'create_player_consistency_features', 
            'create_usage_features', 'create_advanced_offensive_features',
            'create_player_specific_factors', 'engineer_all_features'
        ]:
            if f"_timing_{func_name}" in globals():
                timing_info[func_name] = globals()[f"_timing_{func_name}"]
        
        # Print timing info
        if timing_info:
            logging.info("Feature Engineering Timing Profile:")
            for func_name, time_taken in sorted(timing_info.items(), key=lambda x: x[1], reverse=True):
                logging.info(f"  {func_name}: {time_taken:.2f} seconds")
        else:
            logging.info("No timing information available. Enable @time_function decorators on functions to collect timing data.")