import pandas as pd  # For data manipulation and analysis
import requests     # For making HTTP requests
from bs4 import BeautifulSoup  # For HTML parsing
import time         # For adding delays
import os           # For file operations
import random       # For randomizing delays
import logging      # For better error tracking
from datetime import datetime  # For date handling
import glob         # For finding files with patterns
import argparse     # For command-line argument parsing

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# User agent list for rotation - helps avoid detection as a bot
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:85.0) Gecko/20100101 Firefox/85.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.96 Safari/537.36"
]

# Global session to maintain cookies and connection efficiency
session = requests.Session()

def get_random_user_agent():
    """
    Get a random user agent from the predefined list.
    
    Returns:
        str: A randomly selected user agent string
    """
    return random.choice(USER_AGENTS)

def throttled_request(url, max_retries=5, base_delay=3):
    """
    Make a throttled HTTP request with automatic retries and exponential backoff.
    
    Args:
        url (str): The URL to request
        max_retries (int): Maximum number of retry attempts
        base_delay (int): Base delay between retries in seconds
        
    Returns:
        requests.Response or None: Response object if successful, None otherwise
    """
    # Rotate user agent for each request to avoid detection patterns
    headers = {
        "User-Agent": get_random_user_agent(),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Cache-Control": "max-age=0"
    }
    
    # Add initial random delay (1-3 seconds) to simulate human browsing
    initial_delay = 1 + random.random() * 2
    logging.info(f"Waiting {initial_delay:.2f} seconds before requesting {url}...")
    time.sleep(initial_delay)
    
    # Try making the request with exponential backoff for retries
    for attempt in range(max_retries):
        try:
            logging.info(f"Requesting {url} (attempt {attempt+1}/{max_retries})...")
            response = session.get(url, headers=headers, timeout=30)
            
            # Success case
            if response.status_code == 200:
                return response
                
            # Handle different status codes
            elif response.status_code == 404:  # Not Found - page doesn't exist
                logging.warning(f"Page not found (404): {url}")
                # No need to retry for 404s, just return None
                return None
            elif response.status_code == 429:  # Too Many Requests
                wait_time = base_delay * (2 ** attempt)  # Exponential backoff
                # Add jitter to prevent synchronized requests
                jitter = random.uniform(0.5, 1.5)
                wait_time = wait_time * jitter
                logging.warning(f"Rate limited (429). Waiting {wait_time:.2f} seconds before retry.")
                time.sleep(wait_time)
            elif response.status_code == 403:  # Forbidden
                logging.error(f"Access forbidden (403). The site may have blocked requests.")
                # Add a much longer delay for 403 errors
                time.sleep(60 + random.random() * 60)  # 1-2 minute delay
            else:
                logging.error(f"Failed to retrieve data: Status code: {response.status_code}")
                wait_time = base_delay * (2 ** attempt)
                time.sleep(wait_time)
                
        except requests.exceptions.RequestException as e:
            logging.error(f"Request error: {str(e)}")
            wait_time = base_delay * (2 ** attempt)
            time.sleep(wait_time)
    
    # If all retries fail
    logging.error(f"Max retries exceeded. Could not retrieve data from {url}")
    return None

def scrape_nba_standings():
    """
    Scrape NBA standings data from Basketball Reference website.
    
    Returns:
        dict: Dictionary with dataframes for different standings tables
    """
    # NBA standings URL
    url = "https://www.basketball-reference.com/leagues/NBA_2025_ratings.html"
    
    # Make the throttled request
    response = throttled_request(url)
    if not response:
        return None
    
    # Parse HTML content
    soup = BeautifulSoup(response.content, 'html.parser')

    # Dictionary to store the standings data
    standings_data = {}

    try:
        # Get Team Ratings
        team_ratings_table = soup.find("table", {'id': 'ratings'})
        if team_ratings_table:
            # Use StringIO to avoid FutureWarning
            from io import StringIO
            ratings_html = StringIO(str(team_ratings_table))
            ratings_df = pd.read_html(ratings_html)[0]
            
            # Handle multi-level columns if present
            if isinstance(ratings_df.columns, pd.MultiIndex):
                ratings_df.columns = ratings_df.columns.droplevel(0)
            ratings_df = clean_standings_data(ratings_df)
            
            # Add timestamp for when this data was collected
            ratings_df['Scraped_Date'] = datetime.now().strftime('%Y-%m-%d')
            
            standings_data['team_ratings'] = ratings_df
            logging.info("Successfully scraped Team Ratings")
        
        # Check if any tables were found
        if not standings_data:
            logging.warning("No ratings table found. The website structure might have changed.")
            # List available tables for debugging
            all_tables = soup.find_all('table')
            logging.info(f"Found {len(all_tables)} tables on the page.")
            for i, table in enumerate(all_tables):
                table_id = table.get('id', 'No ID')
                logging.info(f"Table {i+1}: ID = {table_id}")
    
    except Exception as e:
        logging.error(f"Error while scraping data: {str(e)}")
        return None
    
    return standings_data
    
def clean_standings_data(df):
    """
    Clean and format the standings dataframe.
    
    Args:
        df (pandas.DataFrame): Raw standings dataframe
        
    Returns:
        pandas.DataFrame: Cleaned dataframe with formatted columns
    """
    # Remove unnamed columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    # Clean team names (remove asterisks for playoff teams)
    if 'Team' in df.columns:
        df['Team'] = df['Team'].str.replace('*', '', regex=False)
    
    return df

def save_standings_to_csv(standings_data, output_dir='./data'):
    """
    Save the standings data to CSV files with update support for existing files.
    
    Args:
        standings_data (dict): Dictionary of standings dataframes
        output_dir (str): Directory to save the CSV files (default: './data')
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Column mapping for more readable headers
    column_mapping = {
        'MOV': 'Margin_of_Victory',
        'ORtg': 'Offensive_Rating',
        'DRtg': 'Defensive_Rating',
        'NRtg': 'Net_Rating',
        'MOV/A': 'Margin_of_Victory_Adjusted',
        'ORtg/A': 'Offensive_Rating_Adjusted',
        'DRtg/A': 'Defensive_Rating_Adjusted',
        'NRtg/A': 'Net_Rating_Adjusted'
    }

    # Today's date for filename and data tracking
    today_str = datetime.now().strftime('%Y%m%d')
    
    # Save each dataframe to a CSV file
    for name, df in standings_data.items():
        # Create a copy of the dataframe to avoid modifying the original
        df_to_save = df.copy()
        
        # Rename columns for better readability
        for old_col, new_col in column_mapping.items():
            if old_col in df_to_save.columns:
                df_to_save.rename(columns={old_col: new_col}, inplace=True)
        
        # Create output path
        output_path = os.path.join(output_dir, f"{name}_{today_str}.csv")
        
        # Check if file already exists and handle it
        if os.path.exists(output_path):
            # Read existing file
            logging.info(f"Found existing file {output_path}. Updating with new data...")
            existing_df = pd.read_csv(output_path)
            
            # Combine with new data (giving priority to new data)
            # Use 'Team' as the key for merging if it exists
            if 'Team' in existing_df.columns and 'Team' in df_to_save.columns:
                # Remove rows with same teams in existing data (to be replaced with new data)
                existing_teams = df_to_save['Team'].unique()
                existing_df = existing_df[~existing_df['Team'].isin(existing_teams)]
                # Combine old and new data
                df_to_save = pd.concat([existing_df, df_to_save], ignore_index=True)
                logging.info(f"Updated {len(existing_teams)} teams with fresh data.")
            else:
                # If no 'Team' column for merging, just overwrite with new data
                logging.info("No common key for merging. Replacing file with new data.")
            
        # Save to CSV
        df_to_save.to_csv(output_path, index=False)
        logging.info(f"Saved {name} to {output_path}")

def scrape_team_schedules(team_abbrs=None):
    """
    Scrape NBA team schedules and game stats.
    
    Args:
        team_abbrs (dict, optional): Dictionary of team abbreviations to scrape.
                                    If None, all teams will be scraped.
    
    Returns:
        dict: Dictionary with team schedules dataframes
    """
    # Dictionary of team abbreviations if not provided
    if team_abbrs is None:
        team_abbrs = {
            'ATL': 'Atlanta Hawks',
            'BRK': 'Brooklyn Nets',
            'BOS': 'Boston Celtics',
            'CHO': 'Charlotte Hornets',
            'CHI': 'Chicago Bulls',
            'CLE': 'Cleveland Cavaliers',
            'DAL': 'Dallas Mavericks',
            'DEN': 'Denver Nuggets',
            'DET': 'Detroit Pistons',
            'GSW': 'Golden State Warriors',
            'HOU': 'Houston Rockets',
            'IND': 'Indiana Pacers',
            'LAC': 'Los Angeles Clippers',
            'LAL': 'Los Angeles Lakers',
            'MEM': 'Memphis Grizzlies',
            'MIA': 'Miami Heat',
            'MIL': 'Milwaukee Bucks',
            'MIN': 'Minnesota Timberwolves',
            'NOP': 'New Orleans Pelicans',
            'NYK': 'New York Knicks',
            'OKC': 'Oklahoma City Thunder',
            'ORL': 'Orlando Magic',
            'PHI': 'Philadelphia 76ers',
            'PHO': 'Phoenix Suns',
            'POR': 'Portland Trail Blazers',
            'SAC': 'Sacramento Kings',
            'SAS': 'San Antonio Spurs',
            'TOR': 'Toronto Raptors',
            'UTA': 'Utah Jazz',
            'WAS': 'Washington Wizards'
        }
    
    # Dictionary to store team schedules
    team_schedules = {}
    
    # Scrape schedule for each team
    for abbr, team_name in team_abbrs.items():
        try:
            # Construct URL for team schedule
            url = f"https://www.basketball-reference.com/teams/{abbr}/2025_games.html"
            
            # Make throttled request
            response = throttled_request(url, max_retries=4, base_delay=5)
            if not response:
                logging.warning(f"Skipping {team_name} due to request failure.")
                continue
                
            # Parse HTML content
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find the games table
            games_table = soup.find('table', {'id': 'games'})
            
            if games_table:
                # Look specifically for the game_location column
                game_location_th = games_table.find('th', {'data-stat': 'game_location'})
                game_location_exists = game_location_th is not None
                
                # Convert table to dataframe using StringIO to avoid FutureWarning
                from io import StringIO
                games_html = StringIO(str(games_table))
                schedule_df = pd.read_html(games_html)[0]
                
                # Handle multi-level columns if present
                if isinstance(schedule_df.columns, pd.MultiIndex):
                    schedule_df.columns = schedule_df.columns.droplevel(0)
                
                # Check if the game location column was properly captured
                # It might be an unnamed column or have a blank header
                location_col_exists = False
                for col in schedule_df.columns:
                    if col == '' or (isinstance(col, str) and col.isspace()):
                        location_col_exists = True
                        break
                
                # Always extract game location data manually to ensure consistency
                if game_location_exists:
                    logging.info(f"Extracting game location data for {team_name}")
                    # Extract game locations
                    game_locations = []
                    rows = games_table.find('tbody').find_all('tr')
                    for row in rows:
                        # Skip header rows
                        if 'class' in row.attrs and any(c in ' '.join(row.attrs['class']) for c in ['thead', 'divider']):
                            continue
                        
                        # Find the game_location cell
                        location_cell = row.find('td', {'data-stat': 'game_location'})
                        if location_cell:
                            location = location_cell.text.strip()
                            game_locations.append('@' if location == '@' else '')
                        else:
                            game_locations.append('')
                    
                    # Add to dataframe if we have matching length
                    if len(game_locations) == len(schedule_df):
                        schedule_df['Location'] = game_locations
                        logging.info(f"Added {len(game_locations)} game locations for {team_name}")
                    else:
                        logging.warning(f"Game location count mismatch for {team_name}: {len(game_locations)} locations vs {len(schedule_df)} games")
                        # Try to match up rows by date if possible
                        if 'Date' in schedule_df.columns and len(game_locations) > 0:
                            logging.info("Attempting to match game locations by row index")
                            schedule_df['Location'] = ''
                            for i in range(min(len(game_locations), len(schedule_df))):
                                schedule_df.iloc[i, schedule_df.columns.get_loc('Location')] = game_locations[i]
                
                # Clean the dataframe
                schedule_df = clean_schedule_data(schedule_df)
                
                # Add team name and abbreviation columns
                schedule_df['Team'] = team_name
                schedule_df['TeamAbbr'] = abbr
                
                # Add timestamp for when this data was collected
                schedule_df['Scraped_Date'] = datetime.now().strftime('%Y-%m-%d')
                
                # Store in dictionary
                team_schedules[abbr] = schedule_df
                logging.info(f"Successfully scraped schedule for {team_name}")
            else:
                logging.warning(f"No games table found for {team_name}")
        
        except Exception as e:
            logging.error(f"Error scraping {team_name}: {str(e)}")
    
    return team_schedules

def clean_schedule_data(df):
    """
    Clean and format the schedule dataframe.
    
    Args:
        df (pandas.DataFrame): Raw schedule dataframe
        
    Returns:
        pandas.DataFrame: Cleaned dataframe with formatted columns
    """
    # Process the game location column if it exists
    # Check for the unnamed column that contains '@' for away games
    for col in df.columns:
        if col == '' or (isinstance(col, str) and col.isspace()):
            # Rename the unnamed column to 'Location'
            df.rename(columns={col: 'Location'}, inplace=True)
            logging.info("Renamed unnamed game location column to 'Location'")
            break
    
    # Convert location to a clear home/away indicator
    if 'Location' in df.columns:
        # '@' indicates away game, empty or NaN indicates home game
        df['Home'] = df['Location'].apply(lambda x: False if x == '@' else True)
        # Add gameLocation column (1 for home, 0 for away)
        df['gameLocation'] = df['Location'].apply(lambda x: 0 if x == '@' else 1)
    else:
        # If no location column was found, default all games to home (1)
        # This is a fallback and should be rare
        logging.warning("No location column found, defaulting all games to home")
        df['gameLocation'] = 1
    
    # Remove other unnamed columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    
    # Drop rows that are section headers (these typically have 'Date' as NaN)
    if 'Date' in df.columns:
        df = df.dropna(subset=['Date'])
    
    # Convert date strings to datetime objects
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    
    # Clean up score columns - some might be empty for future games
    if 'Tm' in df.columns and 'Opp' in df.columns:
        # Convert score columns to numeric, coercing errors to NaN
        df['Tm'] = pd.to_numeric(df['Tm'], errors='coerce')
        df['Opp'] = pd.to_numeric(df['Opp'], errors='coerce')
    
    # Add a result column (W/L/None for future games)
    if 'Tm' in df.columns and 'Opp' in df.columns:
        # Create a mask for rows where both scores are available
        score_available = df['Tm'].notna() & df['Opp'].notna()
        
        # Initialize Result column with None
        df['Result'] = None
        
        # Set Result based on score comparison
        df.loc[score_available & (df['Tm'] > df['Opp']), 'Result'] = 'W'
        df.loc[score_available & (df['Tm'] < df['Opp']), 'Result'] = 'L'
    
    return df

def save_schedules_to_csv(schedules, output_dir='./data'):
    """
    Save team schedules to CSV files with update support for existing files.
    
    Args:
        schedules (dict): Dictionary of team schedule dataframes
        output_dir (str): Directory to save the CSV files (default: './data')
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Today's date for filename
    today_str = datetime.now().strftime('%Y%m%d')
    
    # Process each team's schedule
    for abbr, schedule_df in schedules.items():
        # Create a copy of the dataframe to avoid modifying the original
        df_to_save = schedule_df.copy()
        
        # Create output path
        output_path = os.path.join(output_dir, f"{abbr}_{today_str}.csv")
        
        # Check if this team's schedule file already exists for today and handle it
        if os.path.exists(output_path):
            logging.info(f"Found existing file {output_path}. Updating with new data...")
            existing_df = pd.read_csv(output_path)
            
            # Convert date column for proper merging
            if 'Date' in existing_df.columns:
                existing_df['Date'] = pd.to_datetime(existing_df['Date'], errors='coerce')
            
            # Merge existing and new data
            # Use 'Date' as the key for identifying unique games
            if 'Date' in existing_df.columns and 'Date' in df_to_save.columns:
                # Select only non-overlapping dates from the existing data
                existing_dates = df_to_save['Date'].dt.date.unique() if hasattr(df_to_save['Date'], 'dt') else []
                existing_df = existing_df[~pd.to_datetime(existing_df['Date']).dt.date.isin(existing_dates)]
                
                # Combine old and new data
                df_to_save = pd.concat([existing_df, df_to_save], ignore_index=True)
                
                # Sort by date
                if 'Date' in df_to_save.columns:
                    df_to_save = df_to_save.sort_values('Date')
                    
                logging.info(f"Updated with fresh data for {len(existing_dates)} game dates.")
            else:
                # If no 'Date' column for merging, just use the new data
                logging.info("No common key for merging. Replacing file with new data.")
        
        # Save to CSV
        df_to_save.to_csv(output_path, index=False)
        logging.info(f"Saved {abbr} schedule to {output_path}")

def find_latest_data_files(pattern, data_dir='./data'):
    """
    Find the latest data files matching a pattern.
    
    Args:
        pattern (str): File pattern to search for (e.g., 'team_ratings_*.csv')
        data_dir (str): Directory to search in
        
    Returns:
        list: List of the latest data files matching the pattern
    """
    # Ensure the data directory exists
    if not os.path.exists(data_dir):
        logging.warning(f"Data directory {data_dir} does not exist.")
        return []
    
    # Create the full search pattern
    search_pattern = os.path.join(data_dir, pattern)
    
    # Find all files matching the pattern
    files = glob.glob(search_pattern)
    
    if not files:
        logging.info(f"No files found matching pattern {search_pattern}")
        return []
    
    # Sort files by modification time (newest first)
    files.sort(key=os.path.getmtime, reverse=True)
    
    return files

def scrape_player_averages(start_season=2015, end_season=2025):
    """
    Scrape NBA player averages per game from Basketball Reference website.
    
    Args:
        start_season (int): Starting season year (default: 2015)
        end_season (int): Ending season year (default: 2025)
        
    Returns:
        dict: Dictionary with player averages dataframes for each season
    """
    player_averages = {}
    
    for season in range(start_season, end_season + 1):
        logging.info(f"Scraping player averages for {season}-{season+1} season...")
        
        # NBA player averages URL
        url = f"https://www.basketball-reference.com/leagues/NBA_{season}_per_game.html"
        
        # Make the throttled request
        response = throttled_request(url, max_retries=5)
        if not response:
            logging.warning(f"Failed to retrieve player averages for {season} season.")
            continue
        
        # Parse HTML content
        soup = BeautifulSoup(response.content, 'html.parser')

        try:
            # Get Player Averages table - first try with the specific ID
            player_stats_table = soup.find("table", {'id': 'per_game_stats'})
            
            if player_stats_table:
                # Use StringIO to avoid FutureWarning
                from io import StringIO
                stats_html = StringIO(str(player_stats_table))
                stats_df = pd.read_html(stats_html)[0]
                
                # Clean the data
                stats_df = clean_player_averages_data(stats_df, season)
                
                # Add to dictionary
                player_averages[season] = stats_df
                logging.info(f"Successfully scraped player averages for {season} season. Found {len(stats_df)} player records.")
            else:
                # Try alternative approach - look for any table with per game stats
                tables = soup.find_all("table")
                found = False
                
                for i, table in enumerate(tables):
                    # Check if this looks like a player stats table
                    if table.find('th', text='PTS') and table.find('th', text='Player'):
                        logging.info(f"Found alternative player stats table (index {i})")
                        from io import StringIO
                        stats_html = StringIO(str(table))
                        stats_df = pd.read_html(stats_html)[0]
                        
                        # Clean the data
                        stats_df = clean_player_averages_data(stats_df, season)
                        
                        # Add to dictionary
                        player_averages[season] = stats_df
                        logging.info(f"Successfully scraped player averages for {season} season. Found {len(stats_df)} player records.")
                        found = True
                        break
                
                if not found:
                    logging.warning(f"No player averages table found for {season} season. The website structure might have changed.")
        
        except Exception as e:
            logging.error(f"Error while scraping player averages for {season} season: {str(e)}")
            continue
    
    return player_averages

def clean_player_averages_data(df, season):
    """
    Clean and format the player averages dataframe.
    
    Args:
        df (pandas.DataFrame): Raw player averages dataframe
        season (int): Season year
        
    Returns:
        pandas.DataFrame: Cleaned dataframe with formatted columns
    """
    # Handle potential header rows - first check if 'Rk' column exists
    if 'Rk' in df.columns:
        # Convert 'Rk' to string first to safely use str methods
        df['Rk'] = df['Rk'].astype(str)
        # Remove header rows that get included as data
        df = df[~df['Rk'].str.contains('Rk', na=False)]
    
    # Handle multi-level columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(0)
    
    # Reset index after removing rows
    df = df.reset_index(drop=True)
    
    # Convert numeric columns to float, but first identify non-numeric columns
    non_numeric_cols = ['Player', 'Pos', 'Tm']
    # Only use columns that actually exist in the dataframe
    existing_non_numeric = [col for col in non_numeric_cols if col in df.columns]
    
    # Convert numeric columns to float
    numeric_cols = df.columns.difference(existing_non_numeric)
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Add season information
    df['Season'] = f"{season}-{str(season+1)[-2:]}"
    df['Season_Year'] = season
    
    # Add timestamp for when this data was collected
    df['Scraped_Date'] = datetime.now().strftime('%Y-%m-%d')
    
    return df

def save_player_averages_to_csv(player_averages, output_dir='./data'):
    """
    Save the player averages data to a single unified CSV file.
    
    Args:
        player_averages (dict): Dictionary of player averages dataframes by season
        output_dir (str): Directory to save the CSV file (default: './data')
    """
    # Create output directory for player stats if it doesn't exist
    player_stats_dir = os.path.join(output_dir, 'player_stats')
    os.makedirs(player_stats_dir, exist_ok=True)
    
    # Today's date for filename
    today_str = datetime.now().strftime('%Y%m%d')
    
    # Combine all seasons into a single dataframe
    if player_averages:
        all_seasons_df = pd.concat(player_averages.values(), ignore_index=True)
        
        # Save the combined dataframe
        output_path = os.path.join(player_stats_dir, f"player_averages_{today_str}.csv")
        all_seasons_df.to_csv(output_path, index=False)
        logging.info(f"Saved unified player averages for all seasons to {output_path}")
        
        # Display sample of the data
        if not all_seasons_df.empty:
            logging.info(f"\nPLAYER AVERAGES SAMPLE (showing first 5 rows):")
            logging.info(all_seasons_df.head())
    else:
        logging.warning("No player averages data to save.")

# Run when script is executed directly
if __name__ == "__main__":
    import argparse
    
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description='NBA Data Scraper - Collect data from Basketball Reference')
    parser.add_argument('--standings', action='store_true', help='Scrape team standings')
    parser.add_argument('--schedules', action='store_true', help='Scrape team schedules')
    parser.add_argument('--players', action='store_true', help='Scrape player averages')
    parser.add_argument('--all', action='store_true', help='Scrape all data (standings, schedules, and player averages)')
    parser.add_argument('--start-season', type=int, default=2015, help='Starting season year for player averages (default: 2015)')
    parser.add_argument('--end-season', type=int, default=2025, help='Ending season year for player averages (default: 2025)')
    parser.add_argument('--output-dir', type=str, default='./data', help='Directory to save data (default: ./data)')
    
    args = parser.parse_args()
    
    # If no specific data type is selected, default to all
    if not (args.standings or args.schedules or args.players):
        args.all = True
    
    logging.info("Starting NBA data scraping...")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Scrape team standings if requested
    if args.standings or args.all:
        logging.info("Scraping team standings...")
        standings_data = scrape_nba_standings()
        if standings_data:
            # Display sample of the data
            logging.info("\nTEAM_RATINGS STANDINGS:")
            if 'team_ratings' in standings_data and not standings_data['team_ratings'].empty:
                logging.info(standings_data['team_ratings'].head())
            
            # Save to CSV
            save_standings_to_csv(standings_data, args.output_dir)
        else:
            logging.warning("Failed to scrape team standings.")
    
    # Scrape team schedules if requested
    if args.schedules or args.all:
        logging.info("\nScraping team schedules...")
        team_schedules = scrape_team_schedules()
        if team_schedules:
            # Save to CSV
            save_schedules_to_csv(team_schedules, args.output_dir)
        else:
            logging.warning("Failed to scrape team schedules.")
    
    # Scrape player averages if requested
    if args.players or args.all:
        logging.info("\nScraping player averages...")
        player_averages = scrape_player_averages(start_season=args.start_season, end_season=args.end_season)
        if player_averages:
            # Save to CSV
            save_player_averages_to_csv(player_averages, args.output_dir)
            
            # Display sample of the data for the first season
            first_season = min(player_averages.keys()) if player_averages else None
            if first_season and not player_averages[first_season].empty:
                logging.info(f"\nPLAYER AVERAGES FOR {first_season} SEASON:")
                logging.info(player_averages[first_season].head())
        else:
            logging.warning("Failed to scrape player averages.")
    
    # Show latest files
    logging.info("\nLatest standings files:")
    latest_standings = find_latest_data_files("team_ratings_*.csv", args.output_dir)
    for file in latest_standings[:3]:  # Show top 3
        logging.info(f" - {os.path.basename(file)}")
    
    logging.info("\nLatest schedule files:")
    latest_schedules = find_latest_data_files("*_[0-9]*.csv", args.output_dir)
    latest_schedules = [f for f in latest_schedules if not f.startswith(os.path.join(args.output_dir, "team_ratings"))]
    latest_schedules = [f for f in latest_schedules if not os.path.dirname(f).endswith("player_stats")]
    for file in latest_schedules[:3]:  # Show top 3
        logging.info(f" - {os.path.basename(file)}")
    
    logging.info("\nLatest player averages files:")
    latest_player_stats = find_latest_data_files("player_averages*.csv", os.path.join(args.output_dir, "player_stats"))
    for file in latest_player_stats[:3]:  # Show top 3
        logging.info(f" - {os.path.basename(file)}")
    
    logging.info("NBA data scraping completed successfully!")