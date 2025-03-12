import requests
import pandas as pd
import os
import csv
import time
import concurrent.futures
import queue
import threading
from datetime import datetime
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables from .env file
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))

# Global rate limiting variables
REQUEST_QUEUE = queue.Queue()
MAX_REQUESTS_PER_MINUTE = 100  # Adjust based on API limits
REQUEST_LOCK = threading.Lock()
LAST_REQUEST_TIMES = []

def rate_limited_request(url, headers, params=None):
    """
    Makes a rate-limited request to prevent exceeding API rate limits.
    
    Args:
        url (str): The URL to request
        headers (dict): Headers to include in the request
        params (dict, optional): Query parameters for the request
        
    Returns:
        requests.Response: The response from the request
    """
    global LAST_REQUEST_TIMES
    
    with REQUEST_LOCK:
        current_time = time.time()
        
        # Remove request timestamps older than 1 minute
        LAST_REQUEST_TIMES = [t for t in LAST_REQUEST_TIMES if current_time - t < 60]
        
        # If we've made too many requests in the last minute, wait
        if len(LAST_REQUEST_TIMES) >= MAX_REQUESTS_PER_MINUTE:
            wait_time = 60 - (current_time - LAST_REQUEST_TIMES[0])
            if wait_time > 0:
                # Silent waiting - no print statement
                time.sleep(wait_time)
                # Recalculate current time after waiting
                current_time = time.time()
        
        # Add current request timestamp
        LAST_REQUEST_TIMES.append(current_time)
    
    # Make the request
    return requests.get(url, headers=headers, params=params)

def fetch_nba_player_list():
    """
    Fetches NBA player list from the Tank01 Fantasy Stats API.
    
    Returns:
        dict: JSON response from the API
    """
    url = "https://tank01-fantasy-stats.p.rapidapi.com/getNBAPlayerList"
    
    # Get API credentials from environment variables
    api_key = os.getenv('RAPIDAPI_KEY')
    api_host = os.getenv('RAPIDAPI_HOST')
    
    if not api_key or not api_host:
        return None
    
    headers = {
        "x-rapidapi-key": api_key,
        "x-rapidapi-host": api_host
    }
    
    try:
        response = rate_limited_request(url, headers)
        response.raise_for_status()  # Raise an exception for HTTP errors
        return response.json()
    except requests.exceptions.RequestException as e:
        return None

def process_player_data(response_data):
    """
    Processes the API response data to extract player IDs and names.
    
    Args:
        response_data (dict): JSON response from the API
        
    Returns:
        list: List of dictionaries containing player IDs and names
    """
    player_data = []
    
    if not response_data or 'body' not in response_data:
        return player_data
    
    # Extract player data from the response
    for player in response_data['body']:
        player_info = {
            'playerID': player.get('playerID', ''),
            'name': player.get('longName', ''),
            'position': player.get('pos', ''),
            'team': player.get('team', ''),
            'teamID': player.get('teamID', '')
        }
        player_data.append(player_info)
    
    return player_data

def save_to_csv(player_data, output_dir):
    """
    Saves player data to a CSV file.
    
    Args:
        player_data (list): List of dictionaries containing player data
        output_dir (str): Directory to save the CSV file
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename with current date
    current_date = datetime.now().strftime("%Y%m%d")
    output_file = os.path.join(output_dir, f"player_info_{current_date}.csv")
    
    # Write data to CSV file
    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            if not player_data:
                return
            
            # Create CSV writer with headers from the first player's keys
            fieldnames = player_data[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            # Write header and data rows
            writer.writeheader()
            writer.writerows(player_data)
    except Exception as e:
        pass

def fetch_player_game_stats(player_id, player_name, season="2025"):
    """
    Fetches individual game statistics for a specific NBA player in a given season.
    
    Args:
        player_id (str): The player ID to fetch game stats for
        player_name (str): The name of the player
        season (str, optional): The season to fetch stats for. Defaults to "2025".
        
    Returns:
        list: List of dictionaries containing game statistics
    """
    url = "https://tank01-fantasy-stats.p.rapidapi.com/getNBAGamesForPlayer"
    
    # Get API credentials from environment variables
    api_key = os.getenv('RAPIDAPI_KEY')
    api_host = os.getenv('RAPIDAPI_HOST')
    
    if not api_key or not api_host:
        return []
    
    headers = {
        "x-rapidapi-key": api_key,
        "x-rapidapi-host": api_host
    }
    
    querystring = {
        "playerID": player_id,
        "season": season,
        "doubleDouble": "0"  # Not filtering for double-doubles
    }
    
    try:
        response = rate_limited_request(url, headers, querystring)
        response.raise_for_status()  # Raise an exception for HTTP errors
        response_data = response.json()
        
        if not response_data or 'body' not in response_data:
            return []
        
        game_stats = []
        
        # Extract game stats from the response
        for game_id, game_data in response_data['body'].items():
            # Skip non-game entries like statusCode
            if game_id == 'statusCode':
                continue
                
            # Add game_id to the stats dictionary
            game_data['gameID'] = game_id
            
            # Ensure player_id and name are included
            game_data['playerID'] = player_id
            game_data['playerName'] = player_name
                
            game_stats.append(game_data)
        
        return game_stats
    except requests.exceptions.RequestException as e:
        return []

def worker_fetch_player_stats(player_chunk, season, result_queue, pbar):
    """
    Worker function to fetch game stats for a chunk of players.
    
    Args:
        player_chunk (list): List of player dictionaries to process
        season (str): The season to fetch stats for
        result_queue (Queue): Queue to store results
        pbar (tqdm): Progress bar to update
    """
    for player in player_chunk:
        player_id = player['playerID']
        player_name = player['name']
        
        # Fetch game stats
        game_stats = fetch_player_game_stats(player_id, player_name, season)
        
        # Add results to queue
        for stat in game_stats:
            result_queue.put(stat)
        
        # Update progress bar
        pbar.update(1)

def save_unified_game_stats(stats_list, season, output_dir):
    """
    Saves all player game statistics to a unified CSV file.
    
    Args:
        stats_list (list): List of dictionaries containing game statistics
        season (str): The season the stats are for
        output_dir (str): Directory to save the CSV file
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename with season
    current_date = datetime.now().strftime("%Y%m%d")
    output_file = os.path.join(output_dir, f"all_player_games_{season}_{current_date}.csv")
    
    # Write data to CSV file
    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            if not stats_list:
                return
            
            # Create CSV writer with headers from the first game's keys
            fieldnames = stats_list[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            # Write header and data rows
            writer.writeheader()
            writer.writerows(stats_list)
    except Exception as e:
        pass

def fetch_all_player_game_stats(player_data_file, season="2025", num_workers=3):
    """
    Fetches game statistics for all players in the provided player data file.
    
    Args:
        player_data_file (str): Path to the CSV file containing player IDs
        season (str, optional): The season to fetch stats for. Defaults to "2025".
        num_workers (int, optional): Number of worker threads. Defaults to 3.
    """
    # Check if file exists
    if not os.path.exists(player_data_file):
        return
    
    # Read player data from CSV
    try:
        player_df = pd.read_csv(player_data_file)
    except Exception as e:
        return
    
    # Check if playerID column exists
    if 'playerID' not in player_df.columns:
        return
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "playerGameStats")
    
    # Convert DataFrame to list of dictionaries
    players = player_df.to_dict('records')
    
    # Create result queue for collecting game stats
    result_queue = queue.Queue()
    
    # Split players into chunks for workers
    chunk_size = max(1, len(players) // num_workers)
    player_chunks = [players[i:i + chunk_size] for i in range(0, len(players), chunk_size)]
    
    # Create progress bar for tracking player processing
    with tqdm(total=len(players), desc="Fetching player game stats", unit="player") as pbar:
        # Create and start worker threads
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit tasks to the executor
            futures = [executor.submit(worker_fetch_player_stats, chunk, season, result_queue, pbar) 
                      for chunk in player_chunks]
            
            # Wait for all futures to complete
            concurrent.futures.wait(futures)
    
    # Collect all results from the queue
    all_game_stats = []
    while not result_queue.empty():
        all_game_stats.append(result_queue.get())
    
    # Save all game stats to a unified CSV file
    save_unified_game_stats(all_game_stats, season, output_dir)

def main():
    """
    Main function to fetch NBA player data and save it to a CSV file.
    """
    print("NBA Game Data Fetcher")
    print("--------------------")
    
    # Fetch NBA player list
    response_data = fetch_nba_player_list()
    
    if not response_data:
        print("Failed to fetch NBA player list. Check your API credentials.")
        return
    
    # Process player data
    player_data = process_player_data(response_data)
    
    if not player_data:
        print("No player data found")
        return
    
    # Save player data to CSV
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "playerInfo")
    save_to_csv(player_data, output_dir)
    
    # Fetch game stats for all players
    current_date = datetime.now().strftime("%Y%m%d")
    player_data_file = os.path.join(output_dir, f"player_info_{current_date}.csv")
    fetch_all_player_game_stats(player_data_file, season="2025", num_workers=3)
    
    print("\nData collection complete!")

if __name__ == "__main__":
    main()