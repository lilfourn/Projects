import os
import pandas as pd
import numpy as np
import glob
import datetime
from pathlib import Path

# Constants
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
SCHEDULES_DIR = os.path.join(DATA_DIR, 'schedules')
RATINGS_FILE = os.path.join(DATA_DIR, 'team_ratings_20250311.csv')
# Directory to save simulation results
SIM_RESULTS_DIR = os.path.join(DATA_DIR, 'simulation_results')
os.makedirs(SIM_RESULTS_DIR, exist_ok=True)

def load_team_ratings(ratings_file=RATINGS_FILE):
    """
    Load team ratings from CSV file.
    
    Args:
        ratings_file: Path to the team ratings CSV file
        
    Returns:
        DataFrame containing team ratings
    """
    try:
        ratings_df = pd.read_csv(ratings_file)
        # Create a mapping of team abbreviations to their ratings
        team_abbr_map = {
            'Atlanta Hawks': 'ATL', 'Boston Celtics': 'BOS', 'Brooklyn Nets': 'BRK',
            'Charlotte Hornets': 'CHO', 'Chicago Bulls': 'CHI', 'Cleveland Cavaliers': 'CLE',
            'Dallas Mavericks': 'DAL', 'Denver Nuggets': 'DEN', 'Detroit Pistons': 'DET',
            'Golden State Warriors': 'GSW', 'Houston Rockets': 'HOU', 'Indiana Pacers': 'IND',
            'Los Angeles Clippers': 'LAC', 'Los Angeles Lakers': 'LAL', 'Memphis Grizzlies': 'MEM',
            'Miami Heat': 'MIA', 'Milwaukee Bucks': 'MIL', 'Minnesota Timberwolves': 'MIN',
            'New Orleans Pelicans': 'NOP', 'New York Knicks': 'NYK', 'Oklahoma City Thunder': 'OKC',
            'Orlando Magic': 'ORL', 'Philadelphia 76ers': 'PHI', 'Phoenix Suns': 'PHO',
            'Portland Trail Blazers': 'POR', 'Sacramento Kings': 'SAC', 'San Antonio Spurs': 'SAS',
            'Toronto Raptors': 'TOR', 'Utah Jazz': 'UTA', 'Washington Wizards': 'WAS'
        }
        
        # Add abbreviation column to ratings
        ratings_df['TeamAbbr'] = ratings_df['Team'].map(lambda x: team_abbr_map.get(x, 'UNK'))
        
        return ratings_df
    except Exception as e:
        print(f"Error loading team ratings: {e}")
        return None

def load_team_schedules(schedules_dir=SCHEDULES_DIR):
    """
    Load all team schedules from CSV files.
    
    Args:
        schedules_dir: Directory containing schedule CSV files
        
    Returns:
        Dictionary mapping team abbreviations to their schedule DataFrames
    """
    schedules = {}
    try:
        # Get all schedule files
        schedule_files = glob.glob(os.path.join(schedules_dir, '*.csv'))
        
        for file_path in schedule_files:
            # Extract team abbreviation from filename (e.g., ATL_20250311.csv -> ATL)
            team_abbr = os.path.basename(file_path).split('_')[0]
            
            # Load schedule
            schedule_df = pd.read_csv(file_path)
            
            # Store in dictionary
            schedules[team_abbr] = schedule_df
        
        return schedules
    except Exception as e:
        print(f"Error loading team schedules: {e}")
        return {}

def predict_game_outcome(home_team_rating, away_team_rating, home_court_advantage=3.0):
    """
    Predict the outcome of a game based on team ratings.
    
    This function uses team ratings and applies a home court advantage to predict
    game outcomes. The gameLocation column in the schedule dataframes (1 for home, 
    0 for away) determines which team gets the home court advantage.
    
    Args:
        home_team_rating: Rating (Net_Rating_Adjusted) of the home team
        away_team_rating: Rating (Net_Rating_Adjusted) of the away team
        home_court_advantage: Rating advantage for home team (default: 3.0)
                             This value is added to the home team's rating
                             to account for home court advantage
        
    Returns:
        Tuple (win_probability, predicted_result)
        where predicted_result is 'W' for home team win, 'L' for home team loss
    """
    # Add home court advantage to home team rating
    adjusted_home_rating = home_team_rating + home_court_advantage
    
    # Calculate win probability using logistic function
    # This converts rating differential to win probability
    rating_diff = adjusted_home_rating - away_team_rating
    win_probability = 1 / (1 + np.exp(-rating_diff * 0.1))
    
    # Determine outcome (deterministic based on probability)
    # In a Monte Carlo simulation, you might use random sampling here
    predicted_result = 'W' if win_probability > 0.5 else 'L'
    
    return win_probability, predicted_result

def get_current_team_record(schedule_df, current_date):
    """
    Get a team's current win-loss record from their schedule.
    
    Args:
        schedule_df: DataFrame with the team's schedule
        current_date: Current date string in format 'YYYY-MM-DD'
        
    Returns:
        Tuple (wins, losses) representing current record
    """
    # Filter to include only games before current date and with a result
    past_games = schedule_df[(pd.to_datetime(schedule_df['Date']) < current_date) & 
                            (pd.notna(schedule_df['Result']))]
    
    if past_games.empty:
        return 0, 0
    
    # Count wins and losses
    wins = len(past_games[past_games['Result'] == 'W'])
    losses = len(past_games[past_games['Result'] == 'L'])
    
    return wins, losses

def calculate_simulation_accuracy(team_schedules, team_ratings):
    """
    Calculate the accuracy of our model by comparing its predictions to actual results
    for games that have already been played.
    
    Args:
        team_schedules: Dictionary mapping team abbreviations to schedule DataFrames
        team_ratings: DataFrame containing team ratings
        
    Returns:
        Tuple (accuracy, correct_predictions, total_games)
    """
    # Get current date
    current_date = datetime.datetime.now().strftime('%Y-%m-%d')
    
    # Create a mapping of team abbreviations to their Net_Rating_Adjusted
    rating_map = {}
    for _, row in team_ratings.iterrows():
        rating_map[row['TeamAbbr']] = row['Net_Rating_Adjusted']
    
    # Create a mapping of team full names to abbreviations
    team_name_to_abbr = {}
    for _, row in team_ratings.iterrows():
        team_name_to_abbr[row['Team']] = row['TeamAbbr']
    
    correct_predictions = 0
    total_games = 0
    
    # Track games to avoid double-counting (since each game appears in two team schedules)
    processed_games = set()
    
    # Store results for detailed reporting
    prediction_results = []
    
    # Process each team's schedule
    for team_abbr, schedule_df in team_schedules.items():
        # Filter to include only games before current date and with a result
        past_games = schedule_df[(pd.to_datetime(schedule_df['Date']) < current_date) & 
                                (pd.notna(schedule_df['Result']))]
        
        # Process each past game
        for _, game in past_games.iterrows():
            # Create a unique identifier for this game to avoid double counting
            game_date = game['Date']
            opponent = game['Opponent']
            opponent_abbr = team_name_to_abbr.get(opponent)
            
            if opponent_abbr is None:
                continue
                
            # Sort team abbreviations to create a consistent game ID
            teams = sorted([team_abbr, opponent_abbr])
            game_id = f"{game_date}_{teams[0]}_{teams[1]}"
            
            # Skip if we've already processed this game
            if game_id in processed_games:
                continue
            
            processed_games.add(game_id)
            
            # Determine if this is a home or away game
            # Use the gameLocation column (1 for home, 0 for away) if available
            if 'gameLocation' in game and pd.notna(game['gameLocation']):
                is_home_game = game['gameLocation'] == 1
            # Fallback to Home column if gameLocation is not available
            elif 'Home' in game and pd.notna(game['Home']):
                is_home_game = bool(game['Home'])
            else:
                # Fallback to our previous heuristic method
                is_home_game = True
                # Method 1: Check Start time - home games typically have exact times
                if pd.isna(game.get('Start (ET)', None)) or not str(game.get('Start (ET)', '')).endswith('p'):
                    is_home_game = False
                    
                # Method 2: Check attendance - home games have attendance numbers
                if pd.isna(game.get('Attend.', 0)) or float(game.get('Attend.', 0)) < 1000:
                    is_home_game = False
                    
                # Method 3: Use game index within schedule as last resort
                if not is_home_game:
                    game_number = int(game.get('G', 0))
                    # Simplistic pattern - in real life might need refinement
                    is_home_game = game_number % 2 == 1
            
            # Get ratings
            team_rating = rating_map.get(team_abbr, 0)
            opponent_rating = rating_map.get(opponent_abbr, 0)
            
            # Predict outcome
            if is_home_game:
                win_prob, predicted_result = predict_game_outcome(team_rating, opponent_rating)
            else:
                win_prob, predicted_result = predict_game_outcome(opponent_rating, team_rating)
                # Flip result for away team perspective
                predicted_result = 'L' if predicted_result == 'W' else 'W'
            
            # Get actual result
            actual_result = game['Result']
            
            # Check if prediction was correct
            is_correct = predicted_result == actual_result
            if is_correct:
                correct_predictions += 1
            
            total_games += 1
            
            # Store detailed results
            prediction_results.append({
                'Date': game_date,
                'Team': team_abbr,
                'Opponent': opponent_abbr,
                'Predicted': predicted_result,
                'Actual': actual_result,
                'Correct': is_correct,
                'Win_Probability': win_prob if predicted_result == 'W' else 1 - win_prob,
                'Home_Game': is_home_game,
                'Game_Number': int(game.get('G', 0))
            })
    
    # Calculate accuracy
    accuracy = correct_predictions / total_games if total_games > 0 else 0
    
    # Convert results to DataFrame for analysis
    results_df = pd.DataFrame(prediction_results)
    
    return accuracy, correct_predictions, total_games, results_df

def simulate_remaining_games(team_schedules, team_ratings):
    """
    Simulate remaining games for all teams.
    
    Args:
        team_schedules: Dictionary mapping team abbreviations to schedule DataFrames
        team_ratings: DataFrame containing team ratings
        
    Returns:
        Dictionary mapping team abbreviations to updated schedule DataFrames with predictions
    """
    # Create a mapping of team abbreviations to their Net_Rating_Adjusted
    rating_map = {}
    for _, row in team_ratings.iterrows():
        rating_map[row['TeamAbbr']] = row['Net_Rating_Adjusted']
    
    # Create a new dictionary to store updated schedules
    updated_schedules = {}
    
    # Current date (to identify remaining games)
    current_date = datetime.datetime.now().strftime('%Y-%m-%d')
    
    # Process each team's schedule
    for team_abbr, schedule_df in team_schedules.items():
        # Create a copy of the schedule
        updated_df = schedule_df.copy()
        
        # Get current team record
        current_wins, current_losses = get_current_team_record(updated_df, current_date)
        
        # Identify future games (Date >= current_date and Result is not set)
        future_games = updated_df[(pd.to_datetime(updated_df['Date']) >= current_date) & 
                                 (updated_df['Result'].isna())]
        
        # Track running win/loss totals for future games
        running_wins = current_wins
        running_losses = current_losses
        
        # Process each future game
        for idx, game in future_games.iterrows():
            # Get opponent abbreviation
            opponent = game['Opponent']
            opponent_abbr = None
            
            # Search for opponent abbreviation in team_ratings
            for _, rating_row in team_ratings.iterrows():
                if rating_row['Team'] == opponent:
                    opponent_abbr = rating_row['TeamAbbr']
                    break
            
            if opponent_abbr is None:
                # If we can't find the opponent, skip this game
                continue
            
            # Determine if this is a home or away game
            # Use the gameLocation column (1 for home, 0 for away) if available
            if 'gameLocation' in game and pd.notna(game['gameLocation']):
                is_home_game = game['gameLocation'] == 1
            # Fallback to Home column if gameLocation is not available
            elif 'Home' in game and pd.notna(game['Home']):
                is_home_game = bool(game['Home'])
            else:
                # Fallback to our previous heuristic method
                is_home_game = True
                # Method 1: Check Start time - home games typically have exact times
                if pd.isna(game.get('Start (ET)', None)) or not str(game.get('Start (ET)', '')).endswith('p'):
                    is_home_game = False
                    
                # Method 2: Check attendance - home games have attendance numbers
                if pd.isna(game.get('Attend.', 0)) or float(game.get('Attend.', 0)) < 1000:
                    is_home_game = False
                    
                # Method 3: Use game index within schedule as last resort
                if not is_home_game:
                    game_number = int(game.get('G', 0))
                    # Simplistic pattern - in real life might need refinement
                    is_home_game = game_number % 2 == 1
            
            # Get ratings
            team_rating = rating_map.get(team_abbr, 0)
            opponent_rating = rating_map.get(opponent_abbr, 0)
            
            # Predict outcome
            if is_home_game:
                win_prob, result = predict_game_outcome(team_rating, opponent_rating)
            else:
                win_prob, result = predict_game_outcome(opponent_rating, team_rating)
                # Flip result for away team perspective
                result = 'L' if result == 'W' else 'W'
            
            # Update running W/L totals
            if result == 'W':
                running_wins += 1
            else:
                running_losses += 1
            
            # Update W/L record in the dataframe
            updated_df.loc[idx, 'W'] = running_wins
            updated_df.loc[idx, 'L'] = running_losses
            
            # Update streak
            if idx > 0:  # Not the first game
                prev_idx = idx - 1
                while prev_idx >= 0 and pd.isna(updated_df.loc[prev_idx, 'Streak']):
                    prev_idx -= 1
                
                if prev_idx >= 0 and pd.notna(updated_df.loc[prev_idx, 'Streak']):
                    prev_streak = str(updated_df.loc[prev_idx, 'Streak'])
                    if len(prev_streak) > 0:
                        prev_result = prev_streak[0]  # W or L
                        # Extract the streak count, handling potential formatting issues
                        try:
                            prev_streak_count = int(''.join(filter(str.isdigit, prev_streak)))
                            if prev_streak_count == 0:  # Fallback if no digits found
                                prev_streak_count = 1
                        except ValueError:
                            # If there's an issue parsing the streak count, default to 1
                            prev_streak_count = 1
                        
                        if prev_result == result:
                            # Continuing streak
                            updated_df.loc[idx, 'Streak'] = f"{result} {prev_streak_count + 1}"
                        else:
                            # New streak
                            updated_df.loc[idx, 'Streak'] = f"{result} 1"
                    else:
                        # Invalid previous streak format
                        updated_df.loc[idx, 'Streak'] = f"{result} 1"
                else:
                    # First streak
                    updated_df.loc[idx, 'Streak'] = f"{result} 1"
            else:
                # First game
                updated_df.loc[idx, 'Streak'] = f"{result} 1"
            
            # Update result
            updated_df.loc[idx, 'Result'] = result
        
        # Store updated schedule
        updated_schedules[team_abbr] = updated_df
    
    return updated_schedules

def calculate_final_standings(updated_schedules, team_ratings):
    """
    Calculate final standings based on simulated schedules.
    
    Args:
        updated_schedules: Dictionary with updated team schedules
        team_ratings: DataFrame with team ratings
        
    Returns:
        DataFrame with final standings
    """
    # Create a list to store final records
    final_records = []
    
    # Process each team
    for team_abbr, schedule_df in updated_schedules.items():
        # Get team name
        team_name = None
        for _, row in team_ratings.iterrows():
            if row['TeamAbbr'] == team_abbr:
                team_name = row['Team']
                break
        
        if team_name is None:
            continue
        
        # Get conference and division
        conf = None
        div = None
        for _, row in team_ratings.iterrows():
            if row['TeamAbbr'] == team_abbr:
                conf = row['Conf']
                div = row['Div']
                break
        
        # Get final record from the last game in the schedule
        final_w = 0
        final_l = 0
        
        if not schedule_df.empty:
            # Filter to only include rows with numeric W and L values
            numeric_rows = schedule_df[schedule_df['W'].apply(lambda x: isinstance(x, (int, float)) or 
                                                           (isinstance(x, str) and x.isdigit()))]
            
            if not numeric_rows.empty:
                # Sort by game number or date to get the last game
                if 'G' in numeric_rows.columns:
                    sorted_df = numeric_rows.sort_values('G')
                elif 'Date' in numeric_rows.columns:
                    sorted_df = numeric_rows.sort_values('Date')
                else:
                    sorted_df = numeric_rows
                
                # Get the last game with valid W/L data
                last_game = sorted_df.iloc[-1]
                
                # Convert W/L to integers
                try:
                    if isinstance(last_game['W'], str) and last_game['W'].isdigit():
                        final_w = int(last_game['W'])
                    elif isinstance(last_game['W'], (int, float)):
                        final_w = int(last_game['W'])
                    
                    if isinstance(last_game['L'], str) and last_game['L'].isdigit():
                        final_l = int(last_game['L'])
                    elif isinstance(last_game['L'], (int, float)):
                        final_l = int(last_game['L'])
                except Exception as e:
                    print(f"Error processing W/L for {team_name}: {e}")
                    print(f"W value: {last_game['W']} (type: {type(last_game['W'])})")
                    print(f"L value: {last_game['L']} (type: {type(last_game['L'])})")
        
        # Calculate win percentage
        win_pct = final_w / (final_w + final_l) if (final_w + final_l) > 0 else 0
        
        # Add to records list
        final_records.append({
            'Team': team_name,
            'TeamAbbr': team_abbr,
            'Conference': conf,
            'Division': div,
            'W': final_w,
            'L': final_l,
            'Win%': round(win_pct, 3)
        })
    
    # Convert to DataFrame and sort by win percentage (descending)
    standings_df = pd.DataFrame(final_records)
    standings_df = standings_df.sort_values(by=['Win%', 'W'], ascending=False)
    
    return standings_df

def save_simulation_results(updated_schedules, standings, sim_results_dir=SIM_RESULTS_DIR):
    """
    Save simulation results to disk.
    
    Args:
        updated_schedules: Dictionary with updated team schedules
        standings: DataFrame with final standings
        sim_results_dir: Directory to save results
    """
    # Save each team's updated schedule
    for team_abbr, schedule_df in updated_schedules.items():
        # Create filename
        current_date = datetime.datetime.now().strftime('%Y%m%d')
        filename = f"{team_abbr}_simulated_{current_date}.csv"
        file_path = os.path.join(sim_results_dir, filename)
        
        # Save to CSV
        schedule_df.to_csv(file_path, index=False)
    
    # Save standings
    standings_path = os.path.join(sim_results_dir, f"final_standings_{datetime.datetime.now().strftime('%Y%m%d')}.csv")
    standings.to_csv(standings_path, index=False)
    
    # Save conference standings
    for conf in ['E', 'W']:
        conf_name = 'Eastern' if conf == 'E' else 'Western'
        conf_standings = standings[standings['Conference'] == conf].reset_index(drop=True)
        conf_standings_path = os.path.join(sim_results_dir, f"{conf_name}_standings_{datetime.datetime.now().strftime('%Y%m%d')}.csv")
        conf_standings.to_csv(conf_standings_path, index=False)

def print_accuracy_details(results_df):
    """
    Print detailed information about prediction accuracy.
    
    Args:
        results_df: DataFrame with prediction results
    """
    # Get accuracy by team
    team_accuracy = results_df.groupby('Team')['Correct'].mean().sort_values(ascending=False)
    
    # Get accuracy by home/away
    home_games = results_df[results_df['Home_Game'] == True]
    away_games = results_df[results_df['Home_Game'] == False]
    home_accuracy = home_games['Correct'].mean() if len(home_games) > 0 else 0
    away_accuracy = away_games['Correct'].mean() if len(away_games) > 0 else 0
    
    # Get accuracy by margin
    results_df['Confidence'] = results_df['Win_Probability'].apply(
        lambda x: 'High' if x > 0.7 else ('Medium' if x > 0.6 else 'Low')
    )
    confidence_accuracy = results_df.groupby('Confidence')['Correct'].agg(['mean', 'count'])
    
    print("\n=== DETAILED ACCURACY REPORT ===")
    
    print("\nTop 5 Teams (Prediction Accuracy):")
    for team, acc in team_accuracy.head(5).items():
        team_games = len(results_df[results_df['Team'] == team])
        print(f"  {team}: {acc:.1%} ({int(acc * team_games)}/{team_games} correct)")
    
    print("\nBottom 5 Teams (Prediction Accuracy):")
    for team, acc in team_accuracy.tail(5).items():
        team_games = len(results_df[results_df['Team'] == team])
        print(f"  {team}: {acc:.1%} ({int(acc * team_games)}/{team_games} correct)")
    
    print(f"\nHome Games: {home_accuracy:.1%} accurate ({len(home_games)} games)")
    print(f"Away Games: {away_accuracy:.1%} accurate ({len(away_games)} games)")
    
    print("\nAccuracy by Prediction Confidence:")
    for confidence, (acc, count) in confidence_accuracy.iterrows():
        print(f"  {confidence} Confidence: {acc:.1%} ({int(acc * count)}/{count} correct)")

def main():
    """
    Main function to run the NBA schedule simulator.
    """
    print("NBA Schedule Simulator Starting...")
    
    # 1. Load team ratings
    print("Loading team ratings...")
    team_ratings = load_team_ratings()
    if team_ratings is None:
        print("Failed to load team ratings. Exiting.")
        return
    
    # 2. Load team schedules
    print("Loading team schedules...")
    team_schedules = load_team_schedules()
    if not team_schedules:
        print("Failed to load team schedules. Exiting.")
        return
    
    # 3. Calculate simulation accuracy for past games
    print("Calculating simulation accuracy for games played to date...")
    accuracy, correct, total, accuracy_details = calculate_simulation_accuracy(team_schedules, team_ratings)
    print(f"\n=== SIMULATION ACCURACY ===")
    print(f"Overall Model Accuracy: {accuracy:.1%} ({correct}/{total} games predicted correctly)")
    print_accuracy_details(accuracy_details)
    
    # 4. Simulate remaining games
    print("\nSimulating remaining games...")
    updated_schedules = simulate_remaining_games(team_schedules, team_ratings)
    
    # 5. Calculate final standings
    print("Calculating final standings...")
    standings = calculate_final_standings(updated_schedules, team_ratings)
    
    # 6. Save results
    print("Saving simulation results...")
    save_simulation_results(updated_schedules, standings)
    
    print(f"Simulation complete! Results saved to {SIM_RESULTS_DIR}")
    
    # Print Eastern Conference standings
    east_standings = standings[standings['Conference'] == 'E'].head(10).reset_index(drop=True)
    print("\nEastern Conference Playoff Picture:")
    print(east_standings[['Team', 'W', 'L', 'Win%']])
    
    # Print Western Conference standings
    west_standings = standings[standings['Conference'] == 'W'].head(10).reset_index(drop=True)
    print("\nWestern Conference Playoff Picture:")
    print(west_standings[['Team', 'W', 'L', 'Win%']])

if __name__ == "__main__":
    main()