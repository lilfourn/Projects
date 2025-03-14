import pandas as pd
import numpy as np
import joblib
import os
import logging
import json
import gc
import glob
from datetime import datetime, timedelta

from src.feature_engineering import engineer_all_features, create_feature_matrix
from src.memory_utils import optimize_dataframe, profile_memory, batch_process, memory_usage_report

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_model(model_path=None):
    """
    Load a trained model from disk
    
    Args:
        model_path (str, optional): Path to the model file. If None, the latest model will be loaded.
        
    Returns:
        object: Loaded model
    """
    if model_path is None:
        # Try to find the latest model file
        models_dir = "/Users/lukesmac/Projects/nbaModel/models"
        if not os.path.exists(models_dir):
            logging.error(f"Models directory not found: {models_dir}")
            return None
        
        # Check for both single joblib models and directory-based models
        model_files = []
        
        # Look for single joblib files first
        single_files = [f for f in os.listdir(models_dir) 
                       if os.path.isfile(os.path.join(models_dir, f)) 
                       and f.startswith("nba_dt_model_") and f.endswith(".joblib")]
        if single_files:
            model_files.extend(single_files)
            
        # Look for directory-based models
        model_dirs = [d for d in os.listdir(models_dir) 
                     if os.path.isdir(os.path.join(models_dir, d)) 
                     and d.startswith("nba_") and "_model_" in d]
        
        if model_dirs:
            # For each model directory, look for joblib files inside
            for model_dir in model_dirs:
                model_dir_path = os.path.join(models_dir, model_dir)
                # Look for joblib files inside (could be nested)
                for root, dirs, files in os.walk(model_dir_path):
                    for file in files:
                        if file.endswith(".joblib"):
                            # Build the relative path from models_dir
                            rel_path = os.path.join(os.path.relpath(root, models_dir), file)
                            model_files.append(rel_path)

        # Sort model files by date, with latest first
        model_files = sorted(model_files, reverse=True)
        
        if not model_files:
            logging.error("No model files found")
            return None
        
        model_path = os.path.join(models_dir, model_files[0])
    
    # Load the model
    try:
        model = joblib.load(model_path)
        logging.info(f"Loaded model from {model_path}")
        return model
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        return None

def load_model_metadata(metadata_path=None):
    """
    Load model metadata (feature and target names)
    
    Args:
        metadata_path (str, optional): Path to the metadata file. If None, the latest metrics file will be used.
        
    Returns:
        tuple: feature_names, target_names
    """
    if metadata_path is None:
        # Try to find the latest metrics file
        models_dir = "/Users/lukesmac/Projects/nbaModel/models"
        if not os.path.exists(models_dir):
            logging.error(f"Models directory not found: {models_dir}")
            return None, None
        
        # Look for both old-style metrics files and new target-specific metrics files
        metrics_files = []
        
        # Look for old-style metrics files
        old_metrics = [f for f in os.listdir(models_dir) 
                       if os.path.isfile(os.path.join(models_dir, f)) 
                       and f.startswith("nba_dt_metrics_") and f.endswith(".json")]
        if old_metrics:
            metrics_files.extend(old_metrics)
            
        # Look for target-specific metrics files (newer format)
        target_metrics = [f for f in os.listdir(models_dir) 
                         if os.path.isfile(os.path.join(models_dir, f)) 
                         and f.startswith("nba_") and "_metrics_" in f and f.endswith(".json")]
        
        if target_metrics:
            metrics_files.extend(target_metrics)
            
        # Sort metrics files with newest first
        metrics_files = sorted(metrics_files, reverse=True)
        
        if not metrics_files:
            logging.error("No metrics files found")
            return None, None
        
        metadata_path = os.path.join(models_dir, metrics_files[0])
    
    # Load the metadata
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        feature_names = metadata.get('feature_names', [])
        target_names = metadata.get('target_names', [])
        
        logging.info(f"Loaded metadata from {metadata_path}")
        return feature_names, target_names
    except Exception as e:
        logging.error(f"Error loading metadata: {str(e)}")
        return None, None

@profile_memory
def prepare_player_data(player_id=None, player_name=None, team=None, opponent=None, use_optimized_types=True):
    """
    Prepare player data for prediction with memory optimization
    
    Args:
        player_id (str, optional): Player ID
        player_name (str, optional): Player name
        team (str, optional): Player's team
        opponent (str, optional): Opponent team
        use_optimized_types (bool): Whether to optimize data types for memory efficiency
        
    Returns:
        pd.DataFrame: DataFrame containing the player data
    """
    # Try to find the latest processed data file
    processed_dir = "/Users/lukesmac/Projects/nbaModel/data/processed"
    if not os.path.exists(processed_dir):
        logging.error(f"Processed data directory not found: {processed_dir}")
        return None
    
    processed_files = sorted(
        [f for f in os.listdir(processed_dir) if f.startswith("processed_nba_data_")],
        reverse=True
    )
    
    if not processed_files:
        logging.error("No processed data files found")
        return None
    
    processed_path = os.path.join(processed_dir, processed_files[0])
    
    # Define column dtypes for optimized loading - only load necessary columns
    dtype_dict = None
    usecols = None
    
    if player_id is not None or player_name is not None:
        # Determine which columns we actually need
        usecols = ['playerID', 'longName', 'teamID', 'teamAbv', 'game_date', 'gameID']
        
        # Add stat columns we'll need
        usecols.extend(['PTS', 'TRB', 'AST', 'STL', 'BLK', 'TOV_y', 'MP', 'Age', 
                       'FG%', '3P%', 'FT%', 'USG%', 'WS/48', 'OBPM', 'DBPM', 'BPM'])
        
        # Add prefix columns
        usecols.extend([col for col in pd.read_csv(processed_path, nrows=1).columns 
                       if col.startswith('std_') or col.startswith('last5_')])
        
        # Define optimized dtypes if requested
        if use_optimized_types:
            dtype_dict = {
                'PTS': 'float32', 'TRB': 'float32', 'AST': 'float32', 'STL': 'float32',
                'BLK': 'float32', 'TOV_y': 'float32', 'MP': 'float32', 'Age': 'float32',
                'FG%': 'float32', '3P%': 'float32', 'FT%': 'float32', 'USG%': 'float32',
                'WS/48': 'float32', 'OBPM': 'float32', 'DBPM': 'float32', 'BPM': 'float32'
            }
            
            # Add dtype for all std_ and last5_ columns
            sample_df = pd.read_csv(processed_path, nrows=1)
            for col in sample_df.columns:
                if col.startswith('std_') or col.startswith('last5_'):
                    dtype_dict[col] = 'float32'
    
    # Load the processed data with optimizations
    try:
        # Only load the columns we need with optimized dtypes
        processed_data = pd.read_csv(processed_path, usecols=usecols, dtype=dtype_dict)
        
        if use_optimized_types:
            # Convert date to proper format more efficiently
            if 'game_date' in processed_data.columns and not pd.api.types.is_datetime64_any_dtype(processed_data['game_date']):
                processed_data['game_date'] = pd.to_datetime(processed_data['game_date'], errors='coerce')
                
            # Convert string columns to categorical for string columns with few unique values
            for col in ['team', 'teamAbv', 'teamID']:
                if col in processed_data.columns and processed_data[col].nunique() < 50:
                    processed_data[col] = processed_data[col].astype('category')
                    
        logging.info(f"Loaded processed data from {processed_path} with shape {processed_data.shape}")
    except Exception as e:
        logging.error(f"Error loading processed data: {str(e)}")
        return None
    
    # Filter for the specific player
    if player_id is not None:
        player_data = processed_data[processed_data['playerID'] == player_id].copy()
    elif player_name is not None:
        # Try to find the player by name (case-insensitive)
        player_data = processed_data[processed_data['longName'].str.lower() == player_name.lower()].copy()
        if len(player_data) == 0:
            # Try partial match
            player_data = processed_data[processed_data['longName'].str.lower().str.contains(player_name.lower())].copy()
    else:
        logging.error("Must provide either player_id or player_name")
        return None
    
    # Clear processed_data from memory as we don't need it anymore
    del processed_data
    gc.collect()
    
    if len(player_data) == 0:
        logging.error(f"No data found for player {player_id or player_name}")
        return None
    
    # Get the most recent game data for the player
    player_data = player_data.sort_values('game_date', ascending=False)
    recent_player_data = player_data.iloc[0].to_dict()
    
    # Create a new row for the upcoming game
    upcoming_game = {}
    
    # Copy over player information
    player_fields = ['playerID', 'longName', 'teamID', 'teamAbv']
    for field in player_fields:
        if field in recent_player_data:
            upcoming_game[field] = recent_player_data[field]
            
    # Set team from teamAbv if available
    if 'teamAbv' in recent_player_data:
        upcoming_game['team'] = recent_player_data['teamAbv']
    
    # Set the team if provided
    if team is not None:
        upcoming_game['team'] = team
        upcoming_game['teamAbv'] = team
    
    # Set the opponent if provided
    if opponent is not None:
        upcoming_game['opponent'] = opponent
        
        # Load opponent strength ratings
        opp_ratings = load_opponent_strength(opponent)
        if opp_ratings is not None:
            # Add opponent strength metrics
            if 'Net_Rating' in opp_ratings:
                upcoming_game['opp_Net_Rating'] = opp_ratings['Net_Rating']
            if 'Offensive_Rating' in opp_ratings:
                upcoming_game['opp_Offensive_Rating'] = opp_ratings['Offensive_Rating']
            if 'Defensive_Rating' in opp_ratings:
                upcoming_game['opp_Defensive_Rating'] = opp_ratings['Defensive_Rating']
            if 'strength_score' in opp_ratings:
                upcoming_game['opp_strength'] = opp_ratings['strength_score']
            else:
                # Default if normalized score wasn't calculated
                upcoming_game['opp_strength'] = 5.0  # Middle of 1-10 scale
        else:
            # Default values if opponent ratings not found
            upcoming_game['opp_Net_Rating'] = 0.0
            upcoming_game['opp_Offensive_Rating'] = 110.0  # League average
            upcoming_game['opp_Defensive_Rating'] = 110.0  # League average
            upcoming_game['opp_strength'] = 5.0  # Middle of 1-10 scale
    
    # Set date to today
    today = datetime.now().strftime("%Y%m%d")
    upcoming_game['game_date'] = today
    upcoming_game['gameID'] = f"{today}_{upcoming_game.get('teamAbv', 'TEAM')}@{opponent or 'OPP'}"
    
    # Use season-to-date stats from the most recent game
    std_fields = [col for col in recent_player_data.keys() if col.startswith('std_')]
    for field in std_fields:
        upcoming_game[field] = recent_player_data[field]
    
    # Use last 5 game averages from the most recent game
    last5_fields = [col for col in recent_player_data.keys() if col.startswith('last5_')]
    for field in last5_fields:
        upcoming_game[field] = recent_player_data[field]
    
    # Copy over season average fields
    season_fields = ['PTS', 'TRB', 'AST', 'STL', 'BLK', 'TOV_y', 'MP', 'Age', 'FG%', '3P%', 'FT%', 'USG%', 'WS/48', 'OBPM', 'DBPM', 'BPM']
    for field in season_fields:
        if field in recent_player_data:
            upcoming_game[field] = recent_player_data[field]
    
    # Create a DataFrame with the upcoming game
    upcoming_df = pd.DataFrame([upcoming_game])
    
    # Clear player_data from memory
    del player_data
    gc.collect()
    
    # Apply data type optimization if requested
    if use_optimized_types:
        datetime_cols = ['game_date']
        upcoming_df = optimize_dataframe(upcoming_df, categorical_threshold=5, 
                                       datetime_cols=datetime_cols, verbose=True)
    
    return upcoming_df

def predict_performance(player_data, model=None, feature_names=None, target_names=None):
    """
    Predict player performance
    
    Args:
        player_data (pd.DataFrame): DataFrame containing player data
        model (object, optional): Trained model. If None, the latest model will be loaded.
        feature_names (list, optional): Feature names. If None, will be loaded from metadata.
        target_names (list, optional): Target names. If None, will be loaded from metadata.
        
    Returns:
        dict: Dictionary of predicted stats
    """
    # Load model and metadata if not provided
    if model is None:
        model = load_model()
        if model is None:
            return None
    
    if feature_names is None or target_names is None:
        feature_names, target_names = load_model_metadata()
        if feature_names is None or target_names is None:
            return None
    
    # Engineer features
    engineered_data = engineer_all_features(player_data)
    
    # Create simplified feature matrix - we'll handle the features manually
    df = engineered_data.copy()
    
    # Get the actual model feature names if available
    model_features = getattr(model, 'feature_names_in_', None)
    if model_features is not None:
        feature_names = model_features
    
    # Create a DataFrame with all required features initialized to 0
    X = pd.DataFrame(0, index=df.index, columns=feature_names)
    
    # Fill in the values we have
    missing_features = []
    for feature in feature_names:
        if feature in df.columns:
            X[feature] = df[feature]
        else:
            missing_features.append(feature)
            # Default values for different feature types
            if 'role_consistency' in feature or 'lineup' in feature or 'chemistry' in feature:
                # Role/lineup features default to median values (0.5)
                X[feature] = 0.5
            else:
                # Other features default to 0
                X[feature] = 0
    
    if missing_features:
        logging.warning(f"Missing {len(missing_features)} features. Some newer features may not be available in this prediction.")
        logging.debug(f"Missing features: {missing_features}")
            
    # Ensure we have no NaN values
    X = X.fillna(0)
    
    # Make predictions
    try:
        y_pred = model.predict(X)
        
        # Convert to dictionary
        predictions = {}
        
        # Handle different output shapes from prediction
        if hasattr(y_pred, 'shape') and len(y_pred.shape) > 1 and y_pred.shape[1] == len(target_names):
            # Multi-target prediction case
            for i, target in enumerate(target_names):
                # Round to appropriate precision
                if target in ['pts', 'reb', 'ast', 'stl', 'blk', 'TOV_x', 'fgm', 'fga', 'tptfgm', 'tptfga']:
                    # Stats like points, rebounds, field goals, etc. are typically whole numbers
                    predictions[target] = round(float(y_pred[0][i]))
                elif target in ['fgp', 'ftp', 'tptfgp']:
                    # Percentages should be rounded to 1 decimal place
                    predictions[target] = round(float(y_pred[0][i]), 1)
                else:
                    # Other stats can be rounded to 2 decimal places
                    predictions[target] = round(float(y_pred[0][i]), 2)
        elif len(target_names) == 1:
            # Single target prediction case
            target = target_names[0]
            
            # Get the predicted value
            if isinstance(y_pred, (list, np.ndarray)):
                pred_value = float(y_pred[0])
            else:
                pred_value = float(y_pred)
                
            # Round to appropriate precision
            if target in ['pts', 'reb', 'ast', 'stl', 'blk', 'TOV_x', 'fgm', 'fga', 'tptfgm', 'tptfga']:
                # Stats like points, rebounds, field goals, etc. are typically whole numbers
                predictions[target] = round(pred_value)
            elif target in ['fgp', 'ftp', 'tptfgp']:
                # Percentages should be rounded to 1 decimal place
                predictions[target] = round(pred_value, 1)
            else:
                # Other stats can be rounded to 2 decimal places
                predictions[target] = round(pred_value, 2)
        else:
            logging.error(f"Prediction output shape {y_pred.shape} doesn't match target names count {len(target_names)}")
            return None
            
        return predictions
    except Exception as e:
        logging.error(f"Error making predictions: {str(e)}")
        return None

def find_next_opponent(team):
    """
    Find the next opponent for a team based on schedule
    
    Args:
        team (str): Team abbreviation
        
    Returns:
        tuple: (opponent_abbr, is_home_game, game_date) or (None, None, None) if not found
    """
    if not team:
        return None
        
    # Ensure team is uppercase for matching
    team = team.upper()
    
    # Map team abbreviations to standard format if needed
    team_mapping = {
        "GSW": "GS",  # Golden State Warriors
        "NOP": "NO",  # New Orleans Pelicans
        "SAS": "SA",  # San Antonio Spurs
        "NYK": "NY",  # New York Knicks
        "LAL": "LAL",  # Los Angeles Lakers (already standard)
        "LAC": "LAC",  # Los Angeles Clippers (already standard)
        "OKC": "OKC",  # Oklahoma City Thunder (already standard)
    }
    
    # Convert to standard format if needed
    standard_team = team_mapping.get(team, team)
    
    # Find the latest schedule file for this team
    schedules_dir = "/Users/lukesmac/Projects/nbaModel/data/schedules"
    if not os.path.exists(schedules_dir):
        logging.warning(f"Schedules directory not found: {schedules_dir}")
        return None
    
    # Find all schedule files for this team
    schedule_pattern = os.path.join(schedules_dir, f"{standard_team}_*.csv")
    schedule_files = sorted(glob.glob(schedule_pattern), reverse=True)
    
    if not schedule_files:
        # Try alternative team abbreviations
        for alt_team, std_team in team_mapping.items():
            if std_team == standard_team:
                continue  # Skip the one we already tried
                
            # Try the alternative abbreviation
            schedule_pattern = os.path.join(schedules_dir, f"{alt_team}_*.csv")
            schedule_files = sorted(glob.glob(schedule_pattern), reverse=True)
            if schedule_files:
                break
    
    if not schedule_files:
        logging.warning(f"No schedule files found for team {team}")
        return None
    
    # Use the most recent schedule file
    schedule_path = schedule_files[0]
    
    try:
        # Load schedule
        schedule = pd.read_csv(schedule_path)
        
        # Get today's date in the same format as the schedule
        today = datetime.now().date()
        
        # Convert date column to datetime
        if 'Date' in schedule.columns:
            schedule['Date'] = pd.to_datetime(schedule['Date']).dt.date
            
            # Find the next game
            future_games = schedule[schedule['Date'] >= today]
            
            if not future_games.empty:
                next_game = future_games.iloc[0]
                
                # Extract opponent
                if 'Opponent' in next_game:
                    opponent_name = next_game['Opponent']
                    
                    # Convert full team name to abbreviation
                    team_name_to_abbr = {
                        "Atlanta Hawks": "ATL",
                        "Boston Celtics": "BOS",
                        "Brooklyn Nets": "BRK",
                        "Charlotte Hornets": "CHO",
                        "Chicago Bulls": "CHI",
                        "Cleveland Cavaliers": "CLE",
                        "Dallas Mavericks": "DAL",
                        "Denver Nuggets": "DEN",
                        "Detroit Pistons": "DET",
                        "Golden State Warriors": "GSW",
                        "Houston Rockets": "HOU",
                        "Indiana Pacers": "IND",
                        "Los Angeles Clippers": "LAC",
                        "Los Angeles Lakers": "LAL",
                        "Memphis Grizzlies": "MEM",
                        "Miami Heat": "MIA",
                        "Milwaukee Bucks": "MIL",
                        "Minnesota Timberwolves": "MIN",
                        "New Orleans Pelicans": "NOP",
                        "New York Knicks": "NYK",
                        "Oklahoma City Thunder": "OKC",
                        "Orlando Magic": "ORL",
                        "Philadelphia 76ers": "PHI",
                        "Phoenix Suns": "PHO",
                        "Portland Trail Blazers": "POR",
                        "Sacramento Kings": "SAC",
                        "San Antonio Spurs": "SAS",
                        "Toronto Raptors": "TOR",
                        "Utah Jazz": "UTA",
                        "Washington Wizards": "WAS"
                    }
                    
                    # Determine if it's a home game
                    is_home = False
                    if 'Home' in next_game and pd.notna(next_game['Home']):
                        is_home = bool(next_game['Home'])
                    elif 'Location' in next_game and pd.notna(next_game['Location']):
                        # If Location is empty or not '@', it's a home game
                        is_home = next_game['Location'] != '@'
                    
                    # Get opponent abbreviation
                    opponent_abbr = team_name_to_abbr.get(opponent_name, opponent_name[:3].upper())
                    
                    # Get game date
                    game_date = next_game['Date'] if isinstance(next_game['Date'], str) else next_game['Date'].strftime('%Y-%m-%d')
                    
                    return opponent_abbr, is_home, game_date
        
        logging.warning(f"Could not find next opponent in schedule for team {team}")
        return None, None, None
    except Exception as e:
        logging.error(f"Error reading schedule file {schedule_path}: {str(e)}")
        return None

def load_opponent_strength(opponent):
    """
    Load opponent strength ratings
    
    Args:
        opponent (str): Opponent team abbreviation
        
    Returns:
        dict: Opponent ratings or None if not found
    """
    if not opponent:
        return None
        
    # Map team abbreviations if needed
    team_mapping = {
        "GSW": "Golden State Warriors",
        "BRK": "Brooklyn Nets",
        "CHO": "Charlotte Hornets",
        "NYK": "New York Knicks",
        "NOP": "New Orleans Pelicans",
        "PHO": "Phoenix Suns",
        "SAS": "San Antonio Spurs",
        "LAL": "Los Angeles Lakers",
        "LAC": "Los Angeles Clippers",
        "OKC": "Oklahoma City Thunder",
    }
    
    # Try to find the latest team ratings file
    standings_dir = "/Users/lukesmac/Projects/nbaModel/data/standings"
    if not os.path.exists(standings_dir):
        logging.warning(f"Standings directory not found: {standings_dir}")
        return None
    
    # Find latest team ratings file
    ratings_files = sorted(
        [f for f in os.listdir(standings_dir) if f.startswith("team_ratings_")],
        reverse=True
    )
    
    if not ratings_files:
        logging.warning("No team ratings files found")
        return None
    
    ratings_path = os.path.join(standings_dir, ratings_files[0])
    
    try:
        # Load ratings
        ratings = pd.read_csv(ratings_path)
        
        # Try to find the opponent team
        if 'Team' in ratings.columns:
            # Try exact match on abbreviation first
            opponent_row = ratings[ratings['Team'] == opponent]
            
            # If not found, try mapping to full name
            if opponent_row.empty and opponent in team_mapping:
                opponent_full = team_mapping[opponent]
                opponent_row = ratings[ratings['Team'] == opponent_full]
            
            # If still not found, try partial match
            if opponent_row.empty:
                for team_name in ratings['Team']:
                    if opponent.lower() in team_name.lower():
                        opponent_row = ratings[ratings['Team'] == team_name]
                        break
            
            if not opponent_row.empty:
                # Convert the row to a dictionary
                opp_ratings = opponent_row.iloc[0].to_dict()
                
                # Add a normalized strength score (1-10 scale, higher is stronger)
                if 'Net_Rating' in opp_ratings:
                    # Get min and max ratings
                    min_rating = ratings['Net_Rating'].min()
                    max_rating = ratings['Net_Rating'].max()
                    
                    # Normalize to 1-10 scale
                    net_rating = opp_ratings['Net_Rating']
                    normalized_rating = 1 + 9 * (net_rating - min_rating) / (max_rating - min_rating)
                    opp_ratings['strength_score'] = normalized_rating
                
                return opp_ratings
        
        logging.warning(f"Could not find ratings for opponent {opponent}")
        return None
    except Exception as e:
        logging.error(f"Error loading team ratings: {str(e)}")
        return None

def predict_player_performance(player_id=None, player_name=None, team=None, opponent=None):
    """
    End-to-end function to predict a player's performance
    
    Args:
        player_id (str, optional): Player ID
        player_name (str, optional): Player name
        team (str, optional): Player's team
        opponent (str, optional): Opponent team
        
    Returns:
        dict: Dictionary containing player info and predicted stats
    """
    # Game environment variables
    is_home_game = None
    game_date = None
    
    # If team is provided but no opponent, try to find next opponent from schedule
    if team is not None and opponent is None:
        opponent_info = find_next_opponent(team)
        if opponent_info[0]:  # If opponent was found
            opponent, is_home_game, game_date = opponent_info
            logging.info(f"Automatically determined next opponent for {player_name}: {opponent} ({'Home' if is_home_game else 'Away'} game on {game_date})")
    
    # Prepare player data
    player_data = prepare_player_data(player_id, player_name, team, opponent)
    
    # Add home game information if available
    if player_data is not None and is_home_game is not None:
        player_data['home_game'] = int(is_home_game)
    if player_data is None:
        return None
        
    # Extract player's recent game stats to calculate recent averages and trends
    # that would normally be calculated in the feature engineering process
    try:
        recent_games = get_player_recent_games(player_id, player_name)
        if recent_games is not None and not recent_games.empty:
            player_data = enrich_with_recent_stats(player_data, recent_games)
    except Exception as e:
        logging.warning(f"Could not enrich player data with recent stats: {str(e)}")
    
    # Get player info
    player_info = {
        'player_id': player_data['playerID'].iloc[0] if 'playerID' in player_data else None,
        'player_name': player_data['longName'].iloc[0] if 'longName' in player_data else player_name,
        'team': team or (player_data['teamAbv'].iloc[0] if 'teamAbv' in player_data else None),
        'opponent': opponent,
        'home_game': player_data['home_game'].iloc[0] if 'home_game' in player_data else None,
        'game_date': game_date
    }
    
    # Load model and metadata
    model = load_model()
    feature_names, target_names = load_model_metadata()
    
    if model is None or feature_names is None or target_names is None:
        return None
    
    # Make predictions
    predictions = predict_performance(player_data, model, feature_names, target_names)
    if predictions is None:
        return None
    
    # Combine player info and predictions
    result = {**player_info, 'predictions': predictions}
    
    return result

def format_prediction_output(predictions):
    """
    Format the prediction output for display
    
    Args:
        predictions (dict): Prediction dictionary
        
    Returns:
        str: Formatted output
    """
    if predictions is None:
        return "No predictions available."
    
    # Extract player info
    player_name = predictions['player_name']
    team = predictions['team']
    opponent = predictions['opponent']
    
    # Extract predictions
    stats = predictions['predictions']
    
    # Get opponent strength info if available
    opponent_info = ""
    if opponent:
        opp_ratings = load_opponent_strength(opponent)
        if opp_ratings is not None and 'strength_score' in opp_ratings:
            strength_level = "Very Strong" if opp_ratings['strength_score'] >= 8 else \
                           "Strong" if opp_ratings['strength_score'] >= 6 else \
                           "Average" if opp_ratings['strength_score'] >= 4 else \
                           "Weak" if opp_ratings['strength_score'] >= 2 else "Very Weak"
            opponent_info = f" ({strength_level} opponent)"
    
    # Get home/away info
    location_info = ""
    if 'home_game' in predictions and predictions['home_game'] is not None:
        location_info = " (Home)" if predictions['home_game'] else " (Away)"
    
    # Format the output
    output = f"Predicted Performance for {player_name} ({team} vs {opponent}{opponent_info}{location_info}):\n"
    output += "-" * 50 + "\n"
    
    # Map stat keys to display names
    stat_display = {
        'pts': 'Points',
        'reb': 'Rebounds',
        'ast': 'Assists',
        'fgm': 'FG Made',
        'fga': 'FG Attempted',
        'tptfgm': '3PT Made',
        'tptfga': '3PT Attempted',
        'stl': 'Steals',
        'blk': 'Blocks',
        'TOV_x': 'Turnovers',
        'fgp': 'FG%',
        'ftp': 'FT%',
        'tptfgp': '3PT%',
        'plusMinus': 'Plus/Minus'
    }
    
    # Format each stat
    for key, display_name in stat_display.items():
        if key in stats:
            if key in ['fgp', 'ftp', 'tptfgp']:
                # Format percentages
                output += f"{display_name}: {stats[key]}%\n"
            else:
                output += f"{display_name}: {stats[key]}\n"
    
    return output

@profile_memory
def predict_multiple_players(player_list, opponent=None, batch_size=10, use_optimized_types=True):
    """
    Predict performance for multiple players using batching for memory efficiency
    
    Args:
        player_list (list): List of player dictionaries with 'name' and optionally 'team'
        opponent (str, optional): Opponent team
        batch_size (int): Number of players to process in each batch
        use_optimized_types (bool): Whether to use optimized data types
        
    Returns:
        list: List of prediction dictionaries
    """
    if not player_list:
        logging.warning("Empty player list provided")
        return []
    
    # Log initial memory usage
    init_mem = memory_usage_report()
    logging.info(f"Starting batch predictions for {len(player_list)} players")
    
    # Load model and metadata once to avoid reloading for each player
    model = load_model()
    feature_names, target_names = load_model_metadata()
    
    if model is None or feature_names is None or target_names is None:
        logging.error("Failed to load model or metadata")
        return []
    
    # Define a batch processing function
    def process_player_batch(batch):
        batch_results = []
        for player in batch:
            player_name = player.get('name')
            team = player.get('team')
            player_opponent = opponent  # Default to the provided opponent
            
            if not player_name:
                logging.warning("Player name not provided, skipping")
                continue
            
            # Game environment variables
            is_home_game = None
            game_date = None
            
            # If no opponent provided, try to determine from schedule
            if player_opponent is None and team is not None:
                opponent_info = find_next_opponent(team)
                if opponent_info[0]:  # If opponent was found
                    player_opponent, is_home_game, game_date = opponent_info
                    logging.info(f"Automatically determined next opponent for {player_name}: {player_opponent} ({'Home' if is_home_game else 'Away'} game on {game_date})")
                
            # Prepare player data
            player_data = prepare_player_data(
                player_name=player_name, 
                team=team, 
                opponent=player_opponent,
                use_optimized_types=use_optimized_types
            )
            
            # Add home game information if available
            if player_data is not None and is_home_game is not None:
                player_data['home_game'] = int(is_home_game)
            
            if player_data is None:
                logging.warning(f"Could not prepare data for player {player_name}")
                continue
                
            # Extract player info
            player_info = {
                'player_id': player_data['playerID'].iloc[0] if 'playerID' in player_data else None,
                'player_name': player_data['longName'].iloc[0] if 'longName' in player_data else player_name,
                'team': team or (player_data['teamAbv'].iloc[0] if 'teamAbv' in player_data else None),
                'opponent': player_opponent,
                'home_game': player_data['home_game'].iloc[0] if 'home_game' in player_data else None,
                'game_date': game_date
            }
            
            # Enrich with recent stats if possible (quietly skip if this fails)
            try:
                recent_games = get_player_recent_games(player_name=player_name)
                if recent_games is not None and not recent_games.empty:
                    player_data = enrich_with_recent_stats(player_data, recent_games)
            except Exception as e:
                logging.debug(f"Could not enrich player data with recent stats: {str(e)}")
            
            # Make predictions using the already loaded model
            try:
                # Engineer features for this player with prediction mode settings
                engineered_data = engineer_all_features(
                    player_data, 
                    use_cache=False, 
                    save_to_cache=False,
                    is_prediction=True,
                    remove_unnecessary_columns=True
                )
                
                # Create feature matrix
                df = engineered_data.copy()
                
                # Debug log player-specific values
                if 'PTS' in df.columns:
                    logging.debug(f"Player {player_name} PTS avg: {df['PTS'].iloc[0]}")
                
                # Create a DataFrame with all required features initialized to 0
                X = pd.DataFrame(0, index=df.index, columns=feature_names)
                
                # Fill in the values we have
                for feature in feature_names:
                    if feature in df.columns:
                        X[feature] = df[feature]
                    else:
                        logging.debug(f"Missing feature: {feature} for player {player_name}")
                
                # Ensure no NaN values
                X = X.fillna(0)
                
                # Log some of the key features for this player
                key_features = ['PTS', 'TRB', 'AST', 'STL', 'BLK'] 
                for feat in key_features:
                    if feat in X.columns:
                        logging.debug(f"{player_name} {feat}: {X[feat].iloc[0]}")
                
                # Predict
                y_pred = model.predict(X)
                
                # Convert to dictionary
                predictions = {}
                for i, target in enumerate(target_names):
                    # Round to appropriate precision
                    if target in ['pts', 'reb', 'ast', 'stl', 'blk', 'TOV_x', 'fgm', 'fga', 'tptfgm', 'tptfga']:
                        # Stats like points, rebounds, field goals, etc. are typically whole numbers
                        predictions[target] = round(float(y_pred[0][i]))
                    elif target in ['fgp', 'ftp', 'tptfgp']:
                        # Percentages should be rounded to 1 decimal place
                        predictions[target] = round(float(y_pred[0][i]), 1)
                    else:
                        # Other stats can be rounded to 2 decimal places
                        predictions[target] = round(float(y_pred[0][i]), 2)
                
                # Combine player info and predictions
                result = {**player_info, 'predictions': predictions}
                batch_results.append(result)
                
                # Clean up memory
                del engineered_data, X, y_pred, predictions
                
            except Exception as e:
                logging.error(f"Error predicting for {player_name}: {str(e)}")
            
            # Clean up this player's data
            del player_data
            gc.collect()
            
        return batch_results
    
    # Process players in batches
    results = []
    
    # Use batch processing
    if batch_size > 1 and len(player_list) > batch_size:
        logging.info(f"Processing {len(player_list)} players in batches of {batch_size}")
        
        # Split the player list into batches and process
        for i in range(0, len(player_list), batch_size):
            batch = player_list[i:i+batch_size]
            logging.info(f"Processing batch {i//batch_size + 1}/{(len(player_list) + batch_size - 1)//batch_size} with {len(batch)} players")
            
            batch_results = process_player_batch(batch)
            results.extend(batch_results)
            
            # Force garbage collection between batches
            gc.collect()
            mem_usage = memory_usage_report()
            logging.info(f"Memory after batch: {mem_usage:.2f} MB")
    else:
        # For a small number of players, process all at once
        results = process_player_batch(player_list)
    
    # Clean up
    del model
    gc.collect()
    
    # Log final memory usage
    final_mem = memory_usage_report()
    logging.info(f"Completed predictions for {len(results)} players. Memory usage: {final_mem:.2f} MB (Change: {final_mem - init_mem:.2f} MB)")
    
    return results

def summarize_team_predictions(predictions):
    """
    Summarize predictions for a team
    
    Args:
        predictions (list): List of player prediction dictionaries
        
    Returns:
        dict: Team summary
    """
    if not predictions:
        return None
    
    # Total team stats
    team_stats = {
        'pts': 0,
        'reb': 0,
        'ast': 0,
        'fgm': 0,
        'fga': 0,
        'tptfgm': 0,
        'tptfga': 0,
        'stl': 0,
        'blk': 0,
        'TOV_x': 0
    }
    
    # Count players with predictions
    player_count = 0
    
    for player in predictions:
        if player and 'predictions' in player:
            player_count += 1
            for stat in team_stats:
                if stat in player['predictions']:
                    team_stats[stat] += player['predictions'][stat]
    
    if player_count == 0:
        return None
    
    # Get team and opponent info from first player
    team = predictions[0]['team'] if predictions and 'team' in predictions[0] else 'TEAM'
    opponent = predictions[0]['opponent'] if predictions and 'opponent' in predictions[0] else 'OPP'
    
    # Create team summary
    summary = {
        'team': team,
        'opponent': opponent,
        'player_count': player_count,
        'stats': team_stats
    }
    
    return summary

def get_player_recent_games(player_id=None, player_name=None, num_games=10):
    """
    Get a player's recent games to calculate trends and averages
    
    Args:
        player_id (str, optional): Player ID
        player_name (str, optional): Player name
        num_games (int, optional): Number of recent games to fetch (default: 10)
        
    Returns:
        pd.DataFrame: DataFrame containing recent game stats
    """
    # Try to find the latest game stats file
    game_stats_dir = "/Users/lukesmac/Projects/nbaModel/data/playerGameStats"
    if not os.path.exists(game_stats_dir):
        logging.error(f"Game stats directory not found: {game_stats_dir}")
        return None
    
    game_stats_files = sorted(
        [f for f in os.listdir(game_stats_dir) if f.startswith("all_player_games_")],
        reverse=True
    )
    
    if not game_stats_files:
        logging.error("No game stats files found")
        return None
    
    game_stats_path = os.path.join(game_stats_dir, game_stats_files[0])
    
    # Load the game stats
    try:
        game_stats = pd.read_csv(game_stats_path)
        logging.info(f"Loaded game stats from {game_stats_path}")
    except Exception as e:
        logging.error(f"Error loading game stats: {str(e)}")
        return None
    
    # Filter for the specific player
    if player_id is not None:
        player_games = game_stats[game_stats['playerID'] == player_id]
    elif player_name is not None:
        # Try to find the player by name (case-insensitive)
        player_games = game_stats[game_stats['longName'].str.lower() == player_name.lower()]
        if len(player_games) == 0:
            # Try partial match
            player_games = game_stats[game_stats['longName'].str.lower().str.contains(player_name.lower())]
    else:
        logging.error("Must provide either player_id or player_name")
        return None
    
    if len(player_games) == 0:
        logging.warning(f"No games found for player {player_id or player_name}")
        return None
    
    # Extract date from gameID and sort by date
    player_games['game_date'] = pd.to_datetime(
        player_games['gameID'].astype(str).str.split('_', expand=True)[0],
        format='%Y%m%d', errors='coerce'
    )
    player_games = player_games.sort_values('game_date', ascending=False)
    
    # Get the most recent games
    recent_games = player_games.head(num_games)
    
    return recent_games

def enrich_with_recent_stats(player_data, recent_games):
    """
    Enrich player data with statistics calculated from recent games
    
    Args:
        player_data (pd.DataFrame): DataFrame containing player data for prediction
        recent_games (pd.DataFrame): DataFrame containing recent game stats
        
    Returns:
        pd.DataFrame: Enriched player data
    """
    # Make a copy to avoid modifying the original
    df = player_data.copy()
    
    # Ensure recent_games has the required columns
    required_cols = ['pts', 'reb', 'ast', 'stl', 'blk', 'TOV_x', 'fgm', 'fga', 'fgp', 
                    'tptfgm', 'tptfga', 'tptfgp', 'ftm', 'fta', 'ftp', 'mins', 'plusMinus']
    
    # Map column names if they're different in recent_games
    col_mapping = {
        'pts': ['pts', 'PTS', 'points'],
        'reb': ['reb', 'TRB', 'rebounds'],
        'ast': ['ast', 'AST', 'assists'],
        'stl': ['stl', 'STL', 'steals'],
        'blk': ['blk', 'BLK', 'blocks'],
        'TOV_x': ['TOV', 'TOV_x', 'turnovers'],
        'fgm': ['fgm', 'FGM', 'field_goals_made'],
        'fga': ['fga', 'FGA', 'field_goals_attempted'],
        'fgp': ['fgp', 'FG%', 'field_goal_percentage'],
        'tptfgm': ['tptfgm', '3PM', 'three_points_made'],
        'tptfga': ['tptfga', '3PA', 'three_points_attempted'],
        'tptfgp': ['tptfgp', '3P%', 'three_point_percentage'],
        'ftm': ['ftm', 'FTM', 'free_throws_made'],
        'fta': ['fta', 'FTA', 'free_throws_attempted'],
        'ftp': ['ftp', 'FT%', 'free_throw_percentage'],
        'mins': ['mins', 'MIN', 'minutes'],
        'plusMinus': ['plusMinus', '+/-', 'plus_minus']
    }
    
    # Create a dictionary to store the mapped columns
    mapped_cols = {}
    for target_col, possible_cols in col_mapping.items():
        for col in possible_cols:
            if col in recent_games.columns:
                mapped_cols[target_col] = col
                break
    
    # Calculate last 3 games averages
    if len(recent_games) >= 1:
        last3_games = recent_games.head(3)
        for target_col, source_col in mapped_cols.items():
            # Convert to numeric, handling any string values
            last3_games[source_col] = pd.to_numeric(last3_games[source_col], errors='coerce')
            avg_value = last3_games[source_col].mean()
            df[f'last3_{target_col}_avg'] = avg_value
    
    # Calculate last 10 games averages
    if len(recent_games) >= 1:
        last10_games = recent_games.head(10)
        for target_col, source_col in mapped_cols.items():
            # Convert to numeric, handling any string values
            last10_games[source_col] = pd.to_numeric(last10_games[source_col], errors='coerce')
            avg_value = last10_games[source_col].mean()
            df[f'last10_{target_col}_avg'] = avg_value
    
    # Calculate trends (last 3 vs last 10)
    for target_col in mapped_cols.keys():
        if f'last3_{target_col}_avg' in df.columns and f'last10_{target_col}_avg' in df.columns:
            df[f'{target_col}_trend'] = df[f'last3_{target_col}_avg'] - df[f'last10_{target_col}_avg']
    
    # Calculate consistency (coefficient of variation)
    if len(recent_games) >= 5:
        for target_col, source_col in mapped_cols.items():
            if target_col in ['pts', 'reb', 'ast', 'mins']:
                # Calculate coefficient of variation (CV)
                std_value = recent_games[source_col].astype(float).std()
                mean_value = recent_games[source_col].astype(float).mean()
                
                if mean_value > 0:
                    cv = std_value / mean_value
                    # Invert so higher values mean more consistency
                    df[f'{target_col}_consistency'] = 1 / (1 + cv)
                else:
                    df[f'{target_col}_consistency'] = 0
    
    return df

def format_team_summary(summary):
    """
    Format team summary for display
    
    Args:
        summary (dict): Team summary dictionary
        
    Returns:
        str: Formatted output
    """
    if summary is None:
        return "No team summary available."
    
    team = summary['team']
    opponent = summary['opponent']
    player_count = summary['player_count']
    stats = summary['stats']
    
    output = f"Team Prediction Summary for {team} vs {opponent} ({player_count} players):\n"
    output += "-" * 50 + "\n"
    
    # Map stat keys to display names
    stat_display = {
        'pts': 'Points',
        'reb': 'Rebounds',
        'ast': 'Assists',
        'fgm': 'FG Made',
        'fga': 'FG Attempted',
        'tptfgm': '3PT Made',
        'tptfga': '3PT Attempted',
        'stl': 'Steals',
        'blk': 'Blocks',
        'TOV_x': 'Turnovers'
    }
    
    # Format each stat
    for key, display_name in stat_display.items():
        if key in stats:
            output += f"Total {display_name}: {stats[key]}\n"
    
    return output

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Predict NBA player performance')
    parser.add_argument('--player-id', type=str, help='Player ID')
    parser.add_argument('--player-name', type=str, help='Player name')
    parser.add_argument('--team', type=str, help='Player team abbreviation')
    parser.add_argument('--opponent', type=str, help='Opponent team abbreviation')
    
    args = parser.parse_args()
    
    if not args.player_id and not args.player_name:
        print("Error: Must provide either player-id or player-name")
        parser.print_help()
        exit(1)
    
    # Predict performance
    prediction = predict_player_performance(
        player_id=args.player_id,
        player_name=args.player_name,
        team=args.team,
        opponent=args.opponent
    )
    
    if prediction:
        print(format_prediction_output(prediction))
    else:
        print("Failed to generate prediction. Check logs for details.")