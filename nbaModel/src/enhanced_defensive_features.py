"""
Enhanced defensive matchup features for NBA prediction models
"""

import numpy as np
import pandas as pd
import os
import logging
from datetime import datetime
import re
import glob
import pickle
from functools import wraps
import time

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
            # Check if caching is explicitly disabled
            use_cache = kwargs.pop('_use_cache', True)
            save_to_cache = kwargs.pop('save_to_cache', True)
            
            # Create cache filename
            cache_file = os.path.join(CACHE_DIR, f"{cache_key}.pkl")
            
            # Check if cache exists and is fresh
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
            
            # Execute function if no cache or cache expired
            result = func(*args, **kwargs)
            
            # Save result to cache
            if save_to_cache:
                try:
                    with open(cache_file, 'wb') as f:
                        logging.info(f"Caching result for {func.__name__} to {cache_file}")
                        pickle.dump(result, f)
                except Exception as e:
                    logging.warning(f"Error writing cache: {str(e)}")
                
            return result
        return wrapper
    return decorator

@time_function
@cache_result('enhanced_defensive_matchup_features')
def create_enhanced_defensive_matchup_features(game_data, team_ratings_path=None, player_stats_path=None):
    """
    Create enhanced defensive matchup features with position-specific defensive metrics
    
    Args:
        game_data (pd.DataFrame): DataFrame containing game data
        team_ratings_path (str, optional): Path to team defensive ratings file
        player_stats_path (str, optional): Path to player season averages
        
    Returns:
        pd.DataFrame: DataFrame with additional defensive matchup features
    """
    # Make a copy of the input DataFrame
    df = game_data.copy()
    
    logging.info("Creating enhanced defensive matchup features...")
    
    # Check if opponent column exists, if not try to extract it from gameID
    if 'opponent' not in df.columns:
        if 'gameID' in df.columns:
            # Extract opponent from gameID (format: YYYYMMDD_TEAM@TEAM or YYYYMMDD_TEAM_TEAM)
            # Parse both formats
            def extract_opponent(game_id, team):
                if '@' in game_id:
                    parts = game_id.split('_')[1].split('@')
                    return parts[1] if parts[0] == team else parts[0]
                else:
                    parts = game_id.split('_')[1:]
                    return parts[1] if parts[0] == team else parts[0]
            
            if 'teamAbv' in df.columns:
                df['opponent'] = df.apply(lambda x: extract_opponent(x['gameID'], x['teamAbv']), axis=1)
            elif 'team' in df.columns:
                df['opponent'] = df.apply(lambda x: extract_opponent(x['gameID'], x['team']), axis=1)
        else:
            logging.warning("Cannot create defensive matchup features: missing opponent information")
            return df
    
    # Add home/away game indicator if not present
    if 'home_game' not in df.columns and 'gameID' in df.columns:
        # Determine home/away based on gameID format
        def is_home_game(game_id, team):
            if '@' in game_id:
                parts = game_id.split('_')[1].split('@')
                return parts[0] == team  # Format: YYYYMMDD_HOME@AWAY
            return False  # Default if format is unknown
        
        if 'teamAbv' in df.columns:
            df['home_game'] = df.apply(lambda x: is_home_game(x['gameID'], x['teamAbv']), axis=1).astype(int)
        elif 'team' in df.columns:
            df['home_game'] = df.apply(lambda x: is_home_game(x['gameID'], x['team']), axis=1).astype(int)
    
    # Load team defensive ratings if path provided
    team_ratings = None
    if team_ratings_path is None:
        # Try to find latest team ratings file
        ratings_dir = os.path.join(DATA_DIR, "standings")
        if os.path.exists(ratings_dir):
            rating_files = sorted([f for f in os.listdir(ratings_dir) if f.startswith("team_ratings_")], reverse=True)
            if rating_files:
                team_ratings_path = os.path.join(ratings_dir, rating_files[0])
    
    if team_ratings_path and os.path.exists(team_ratings_path):
        try:
            team_ratings = pd.read_csv(team_ratings_path)
            logging.info(f"Loaded team ratings from {team_ratings_path}")
            
            # Expected columns: 'Team', 'Team_Abbr', 'W', 'L', 'Win_PCT', 'MIN', 'ORtg', 'DRtg', 'Net_Rating', 'Pace'
            rating_cols = ['Defensive_Rating', 'DRtg', 'OPP_PTS', 'OPP_FG_PCT', 'OPP_3P_PCT', 
                          'DEF_RATING', 'BLK', 'STL', 'OPP_TOV']
            
            # Use available columns
            available_rating_cols = [col for col in rating_cols if col in team_ratings.columns]
            
            if not available_rating_cols:
                logging.warning(f"No defensive rating columns found in {team_ratings_path}")
                team_ratings = None
        except Exception as e:
            logging.error(f"Error loading team ratings: {str(e)}")
            team_ratings = None
    
    # Step 1: Load defensive player data or calculate from averages
    player_defensive_stats = {}
    position_mapping = {
        'PG': 1, 'SG': 2, 'SF': 3, 'PF': 4, 'C': 5,
        'G': 1.5, 'F': 3.5, 'G-F': 2.5, 'F-G': 2.5, 'F-C': 4.5, 'C-F': 4.5
    }
    
    # Load player season averages if path provided
    player_avg = None
    if player_stats_path is None:
        # Try to find latest player stats file
        player_dir = os.path.join(DATA_DIR, "player_stats")
        if os.path.exists(player_dir):
            player_files = sorted([f for f in os.listdir(player_dir) if f.startswith("player_averages_")], reverse=True)
            if player_files:
                player_stats_path = os.path.join(player_dir, player_files[0])
    
    # Try to load player info file which includes positions
    player_info = None
    player_info_dir = os.path.join(DATA_DIR, "playerInfo")
    if os.path.exists(player_info_dir):
        info_files = sorted([f for f in os.listdir(player_info_dir) if f.startswith("player_info_")], reverse=True)
        if info_files:
            try:
                player_info_path = os.path.join(player_info_dir, info_files[0])
                player_info = pd.read_csv(player_info_path)
                logging.info(f"Loaded player info from {player_info_path}")
            except Exception as e:
                logging.error(f"Error loading player info: {str(e)}")
                player_info = None
    
    # Load player averages
    if player_stats_path and os.path.exists(player_stats_path):
        try:
            player_avg = pd.read_csv(player_stats_path)
            logging.info(f"Loaded player averages from {player_stats_path}")
            
            # Merge with player info if available to get positions
            if player_info is not None and 'playerID' in player_info.columns and 'playerID' in player_avg.columns:
                # Join on playerID
                player_avg = pd.merge(player_avg, player_info[['playerID', 'position']], on='playerID', how='left')
            
            # Fill missing positions with a default
            if 'position' not in player_avg.columns:
                player_avg['position'] = 'U'  # Unknown position
            
            # Calculate defensive metrics for each player
            # Key defensive stats: STL, BLK, defensive_rating (if available)
            # Build a dictionary of player defensive metrics
            for _, row in player_avg.iterrows():
                player_id = row.get('playerID', None)
                if player_id:
                    # Extract position and convert to numeric value
                    position = row.get('position', 'U')
                    position_value = position_mapping.get(position, 3)  # Default to SF (3) if unknown
                    
                    # Calculate defensive metrics
                    defensive_metrics = {
                        'stl': row.get('STL', 0),
                        'blk': row.get('BLK', 0),
                        'def_rtg': row.get('defensive_rating', 110),  # Default if not available
                        'position': position,
                        'position_value': position_value
                    }
                    
                    # Add additional stats if available
                    for stat in ['dws', 'dbpm', 'drb']:
                        if stat.upper() in row:
                            defensive_metrics[stat] = row[stat.upper()]
                    
                    player_defensive_stats[player_id] = defensive_metrics
            
            logging.info(f"Created defensive metrics for {len(player_defensive_stats)} players")
            
        except Exception as e:
            logging.error(f"Error loading player averages: {str(e)}")
            player_avg = None
    
    # Step 2: Create team-level defensive metrics
    team_defensive_metrics = {}
    
    if team_ratings is not None:
        # Normalize defensive ratings to 0-1 scale (0 = worst defense, 1 = best defense)
        if 'DRtg' in team_ratings.columns:
            # Note: Lower DRtg is better, so we need to invert the scale
            min_drtg = team_ratings['DRtg'].min()
            max_drtg = team_ratings['DRtg'].max()
            
            if max_drtg > min_drtg:
                for _, row in team_ratings.iterrows():
                    team_abbr = row['Team_Abbr'] if 'Team_Abbr' in row else row['Team']
                    drtg = row['DRtg']
                    # Invert scale: best defense (lowest DRtg) = 1
                    normalized_defense = (max_drtg - drtg) / (max_drtg - min_drtg)
                    
                    # Store team metrics
                    team_defensive_metrics[team_abbr] = {
                        'def_rating': drtg,
                        'def_strength': normalized_defense
                    }
                    
                    # Add position-specific metrics if available
                    if 'ORtg_vs_PG' in row:
                        team_defensive_metrics[team_abbr].update({
                            'def_vs_pg': (100 - row['ORtg_vs_PG'])/100,
                            'def_vs_sg': (100 - row['ORtg_vs_SG'])/100,
                            'def_vs_sf': (100 - row['ORtg_vs_SF'])/100,
                            'def_vs_pf': (100 - row['ORtg_vs_PF'])/100,
                            'def_vs_c': (100 - row['ORtg_vs_C'])/100
                        })
    
    # Step 3: Create player position data if not available directly
    if player_avg is not None and ('position' not in player_avg.columns or player_avg['position'].isna().all()):
        # Try to infer positions from player stats
        for _, row in player_avg.iterrows():
            player_id = row.get('playerID', None)
            if player_id and player_id not in player_defensive_stats:
                # Heuristic position inference based on stats
                pts = row.get('PTS', 0)
                reb = row.get('TRB', 0)
                ast = row.get('AST', 0)
                stl = row.get('STL', 0)
                blk = row.get('BLK', 0)
                height = row.get('HEIGHT', 0) if 'HEIGHT' in row else 0
                
                # Simple position inference based on stats
                if height > 82 and reb > 7:  # Tall with lots of rebounds
                    position = 'C'
                    position_value = 5
                elif height > 80 and reb > 6:  # Tall with decent rebounds
                    position = 'PF'
                    position_value = 4
                elif height > 78 and pts > 15:  # Medium height, good scoring
                    position = 'SF'
                    position_value = 3
                elif ast > 5:  # High assists
                    position = 'PG'
                    position_value = 1
                elif pts > 15 and height < 78:  # Shorter scorer
                    position = 'SG'
                    position_value = 2
                else:
                    position = 'F'  # Default to forward
                    position_value = 3.5
                
                # Create entry
                player_defensive_stats[player_id] = {
                    'stl': stl,
                    'blk': blk,
                    'def_rtg': 110,  # Default
                    'position': position,
                    'position_value': position_value
                }
    
    # Step 4: Create position matchup advantages
    # Now incorporate position matchup information into the player data
    if 'position' in df.columns:
        # We have position data directly in the game data
        position_col = 'position'
    elif 'pos' in df.columns:
        position_col = 'pos'
    else:
        # Try to add position by joining with player info
        if player_info is not None and 'playerID' in df.columns:
            df = pd.merge(df, player_info[['playerID', 'position']], on='playerID', how='left')
            position_col = 'position' if 'position' in df.columns else None
        else:
            position_col = None
    
    # Create position value column if we have position data
    if position_col:
        df['position_value'] = df[position_col].map(position_mapping).fillna(3)
    elif 'playerID' in df.columns:
        # Use the dictionary we built earlier
        df['position_value'] = df['playerID'].map(lambda x: player_defensive_stats.get(x, {}).get('position_value', 3))
    else:
        # No position data available
        df['position_value'] = 3  # Default to SF
    
    # Step 5: Add team defensive metrics to the dataframe
    if team_defensive_metrics:
        # Add team defensive metrics based on opponent
        df['opp_def_rating'] = df['opponent'].map(lambda x: team_defensive_metrics.get(x, {}).get('def_rating', 110))
        df['opp_def_strength'] = df['opponent'].map(lambda x: team_defensive_metrics.get(x, {}).get('def_strength', 0.5))
        
        # Add position-specific defensive metrics
        def get_pos_def_metric(row):
            opp = row['opponent']
            pos_val = row['position_value']
            team_metrics = team_defensive_metrics.get(opp, {})
            
            # Interpolate between position values
            if pos_val <= 1.5:  # Between PG and SG
                pg_def = team_metrics.get('def_vs_pg', 0.5)
                sg_def = team_metrics.get('def_vs_sg', 0.5)
                weight_sg = pos_val - 1
                return pg_def * (1 - weight_sg) + sg_def * weight_sg
            elif pos_val <= 2.5:  # Between SG and SF
                sg_def = team_metrics.get('def_vs_sg', 0.5)
                sf_def = team_metrics.get('def_vs_sf', 0.5)
                weight_sf = pos_val - 2
                return sg_def * (1 - weight_sf) + sf_def * weight_sf
            elif pos_val <= 3.5:  # Between SF and PF
                sf_def = team_metrics.get('def_vs_sf', 0.5)
                pf_def = team_metrics.get('def_vs_pf', 0.5)
                weight_pf = pos_val - 3
                return sf_def * (1 - weight_pf) + pf_def * weight_pf
            elif pos_val <= 4.5:  # Between PF and C
                pf_def = team_metrics.get('def_vs_pf', 0.5)
                c_def = team_metrics.get('def_vs_c', 0.5)
                weight_c = pos_val - 4
                return pf_def * (1 - weight_c) + c_def * weight_c
            else:  # C or beyond
                return team_metrics.get('def_vs_c', 0.5)
        
        # Add position-specific defensive strength
        if any('def_vs_pg' in team_metrics for team_metrics in team_defensive_metrics.values()):
            df['pos_specific_def'] = df.apply(get_pos_def_metric, axis=1)
        else:
            # If position-specific metrics aren't available, use overall team defense
            df['pos_specific_def'] = df['opp_def_strength']
    else:
        # Default values if no team metrics available
        df['opp_def_rating'] = 110
        df['opp_def_strength'] = 0.5
        df['pos_specific_def'] = 0.5
    
    # Step 6: Create defensive matchup advantage metrics
    # Calculate position matchup advantages - how well does this player's position
    # typically perform against the opposing team's defense
    
    # Create features for different statistical categories
    stat_categories = ['pts', 'reb', 'ast', 'stl', 'blk', 'fgm', 'tptfgm']
    
    for stat in stat_categories:
        # Create matchup advantage metric for each stat type (scaled 0-1)
        # Higher value = more favorable matchup for the player
        if f'pos_specific_def' in df.columns:
            # Inverse of defensive strength (1 - strength) since
            # higher defensive strength means worse matchup for offense
            df[f'def_matchup_{stat}'] = 1 - df['pos_specific_def']
        else:
            # Use overall team defense as fallback
            df[f'def_matchup_{stat}'] = 1 - df['opp_def_strength']
    
    # Step 7: Calculate recent defensive performance against similar players
    # This would ideally use recent game logs against players of similar position/role
    # For simplicity, we'll use a general adjustment based on position
    
    # Adjust based on recent performances against position
    # (Advanced version would analyze actual recent game logs against position)
    adjustments = {
        'pts': {'PG': 0.02, 'SG': 0.01, 'SF': 0, 'PF': -0.01, 'C': -0.02},
        'reb': {'PG': -0.03, 'SG': -0.02, 'SF': 0, 'PF': 0.02, 'C': 0.03},
        'ast': {'PG': 0.03, 'SG': 0.01, 'SF': 0, 'PF': -0.02, 'C': -0.03},
        'blk': {'PG': -0.04, 'SG': -0.03, 'SF': -0.01, 'PF': 0.02, 'C': 0.04},
        'stl': {'PG': 0.03, 'SG': 0.02, 'SF': 0.01, 'PF': -0.01, 'C': -0.02},
        'fgm': {'PG': 0.01, 'SG': 0.01, 'SF': 0, 'PF': 0, 'C': -0.01},
        'tptfgm': {'PG': 0.03, 'SG': 0.02, 'SF': 0.01, 'PF': -0.01, 'C': -0.03}
    }
    
    if position_col:
        for stat, pos_adjustments in adjustments.items():
            # Apply position-specific adjustments
            for pos, adj in pos_adjustments.items():
                # Apply adjustment where position matches
                mask = df[position_col] == pos
                if mask.any() and f'def_matchup_{stat}' in df.columns:
                    df.loc[mask, f'def_matchup_{stat}'] += adj
            
            # Ensure values stay in 0-1 range
            if f'def_matchup_{stat}' in df.columns:
                df[f'def_matchup_{stat}'] = df[f'def_matchup_{stat}'].clip(0, 1)
    
    # Step 8: Apply home court defensive adjustment
    # Teams typically defend better at home
    if 'home_game' in df.columns:
        home_def_boost = 0.02  # 2% boost to defensive metrics for home team
        
        # Identify situations where player's team is away (opponent is home)
        # In these cases, the opposing defense gets stronger
        away_mask = df['home_game'] == 0
        
        for stat in stat_categories:
            if f'def_matchup_{stat}' in df.columns:
                # Away games: opponent defense is stronger, so matchup is less favorable
                df.loc[away_mask, f'def_matchup_{stat}'] -= home_def_boost
                
                # Ensure values stay in 0-1 range
                df[f'def_matchup_{stat}'] = df[f'def_matchup_{stat}'].clip(0, 1)
    
    # Step 9: Create defensive versatility metrics
    # How well teams defend different types of plays (scoring, rebounding, passing)
    # This is a more advanced concept that would require play-by-play data
    # For now, we'll implement a simple version based on available team metrics
    
    if team_defensive_metrics:
        # Create metrics for how well a team defends different play types
        scoring_defense = {}
        rebounding_defense = {}
        playmaking_defense = {}
        
        for team, metrics in team_defensive_metrics.items():
            # Example formulation (would be better with actual play defense data)
            base_def = metrics.get('def_strength', 0.5)
            
            # Teams that defend guards well typically defend passing well
            guard_def = (metrics.get('def_vs_pg', 0.5) + metrics.get('def_vs_sg', 0.5)) / 2
            
            # Teams that defend bigs well typically defend rebounding well
            big_def = (metrics.get('def_vs_pf', 0.5) + metrics.get('def_vs_c', 0.5)) / 2
            
            # For scoring, use overall defensive rating
            scoring_defense[team] = base_def
            
            # For rebounding, weight big defense more heavily
            rebounding_defense[team] = 0.7 * big_def + 0.3 * base_def
            
            # For playmaking, weight guard defense more heavily
            playmaking_defense[team] = 0.7 * guard_def + 0.3 * base_def
        
        # Add these metrics to the dataframe
        df['opp_scoring_def'] = df['opponent'].map(lambda x: scoring_defense.get(x, 0.5))
        df['opp_rebounding_def'] = df['opponent'].map(lambda x: rebounding_defense.get(x, 0.5))
        df['opp_playmaking_def'] = df['opponent'].map(lambda x: playmaking_defense.get(x, 0.5))
        
        # Convert to matchup advantage (inverse of defensive strength)
        df['scoring_matchup'] = 1 - df['opp_scoring_def']
        df['rebounding_matchup'] = 1 - df['opp_rebounding_def']
        df['playmaking_matchup'] = 1 - df['opp_playmaking_def']
    
    # Step 10: Create advanced matchup performance indices
    # These are composite indices that combine defensive matchups with player abilities
    
    # For these indices, we need both player offensive abilities and opponent defensive metrics
    # Higher values indicate better expected performance
    
    # Scoring Index - combines player scoring ability with opponent's defensive matchup
    if all(col in df.columns for col in ['PTS', 'def_matchup_pts']):
        # Normalize PTS to 0-1 scale
        max_pts = df['PTS'].max()
        if max_pts > 0:
            normalized_pts = df['PTS'] / max_pts
            df['scoring_index'] = (normalized_pts * 0.7 + df['def_matchup_pts'] * 0.3)
    
    # Rebounding Index
    if all(col in df.columns for col in ['TRB', 'def_matchup_reb']):
        max_reb = df['TRB'].max()
        if max_reb > 0:
            normalized_reb = df['TRB'] / max_reb
            df['rebounding_index'] = (normalized_reb * 0.7 + df['def_matchup_reb'] * 0.3)
    
    # Assist Index
    if all(col in df.columns for col in ['AST', 'def_matchup_ast']):
        max_ast = df['AST'].max()
        if max_ast > 0:
            normalized_ast = df['AST'] / max_ast
            df['assist_index'] = (normalized_ast * 0.7 + df['def_matchup_ast'] * 0.3)
    
    # Return the enhanced dataframe
    logging.info(f"Created enhanced defensive matchup features with {len(df)} rows")
    return df