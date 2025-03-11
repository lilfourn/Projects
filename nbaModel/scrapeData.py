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
import sys
import traceback

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

def scrape_standings(start_season=2023, end_season=2024):
    """
    Scrape NBA standings data from Basketball Reference website.
    
    Args:
        start_season (int): Starting season year (default: 2023)
        end_season (int): Ending season year (default: 2024)
        
    Returns:
        dict: Dictionary with dataframes for different standings tables
    """
    standings_data = {}
    
    for season in range(start_season, end_season + 1):
        logging.info(f"Scraping standings data for {season}-{season+1} season...")
        
        # NBA standings URL
        url = f"https://www.basketball-reference.com/leagues/NBA_{season}_standings.html"
        
        # Make the throttled request
        response = throttled_request(url)
        if not response:
            logging.warning(f"Failed to retrieve standings data for {season} season.")
            continue
        
        # Parse HTML content
        soup = BeautifulSoup(response.content, 'html.parser')

        try:
            # Get Team Ratings
            team_ratings_table = soup.find("table", {'id': 'standings'})
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
                
                standings_data[f"{season}-{season+1}"] = ratings_df
                logging.info(f"Successfully scraped standings for {season}-{season+1} season.")
        
        except Exception as e:
            logging.error(f"Error while scraping standings data for {season} season: {str(e)}")
            continue
    
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
                    rows = games_table.find_all('tr')
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
                
                # Extract team abbreviations directly from HTML since pandas read_html doesn't preserve data-stat attributes
                team_abbrs = []
                rows = player_stats_table.find_all('tr')
                
                for row in rows:
                    # Skip header rows
                    if row.get('class') and 'thead' in row.get('class'):
                        continue
                    
                    # Find the team abbreviation cell (data-stat="team_name_abbr")
                    team_cell = row.find('td', {'data-stat': 'team_id'}) or row.find('td', {'data-stat': 'team_name_abbr'})
                    if team_cell:
                        team_abbrs.append(team_cell.text.strip())
                    else:
                        team_abbrs.append(None)  # Add None if not found
                
                # Add team abbreviations to dataframe if we have the right number
                if len(team_abbrs) == len(stats_df):
                    stats_df['Team_Abbr'] = team_abbrs
                
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
                        
                        # Extract team abbreviations directly from HTML
                        team_abbrs = []
                        rows = table.find_all('tr')
                        
                        for row in rows:
                            # Skip header rows
                            if row.get('class') and 'thead' in row.get('class'):
                                continue
                            
                            # Find the team abbreviation cell (data-stat="team_name_abbr")
                            team_cell = row.find('td', {'data-stat': 'team_id'}) or row.find('td', {'data-stat': 'team_name_abbr'})
                            if team_cell:
                                team_abbrs.append(team_cell.text.strip())
                            else:
                                team_abbrs.append(None)  # Add None if not found
                        
                        # Add team abbreviations to dataframe if we have the right number
                        if len(team_abbrs) == len(stats_df):
                            stats_df['Team_Abbr'] = team_abbrs
                        
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
    
    # Handle team column - prioritize our custom Team_Abbr if it exists
    if 'Team_Abbr' in df.columns:
        # Rename to standard column name
        df['Tm'] = df['Team_Abbr']
        df = df.drop(columns=['Team_Abbr'])
    elif 'Tm' not in df.columns and 'Team' in df.columns:
        # If we have 'Team' but not 'Tm', rename it
        df = df.rename(columns={'Team': 'Tm'})
    
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

def scrape_player_per36_minutes(start_season=2015, end_season=2025):
    """
    Scrape NBA player averages per 36 minutes from Basketball Reference website.
    
    Args:
        start_season (int): Starting season year (default: 2015)
        end_season (int): Ending season year (default: 2025)
        
    Returns:
        dict: Dictionary with player per 36 minutes dataframes for each season
    """
    player_per36 = {}
    
    for season in range(start_season, end_season + 1):
        logging.info(f"Scraping player per 36 minutes for {season}-{season+1} season...")
        
        # NBA player per 36 minutes URL
        url = f"https://www.basketball-reference.com/leagues/NBA_{season}_per_minute.html"
        
        # Make the throttled request
        response = throttled_request(url, max_retries=5)
        if not response:
            logging.warning(f"Failed to retrieve player per 36 minutes for {season} season.")
            continue
        
        # Parse HTML content
        soup = BeautifulSoup(response.content, 'html.parser')

        try:
            # Get Player Per 36 Minutes table - first try with the specific ID
            player_stats_table = soup.find("table", {'id': 'per_minute_stats'})
            
            if player_stats_table:
                # Use StringIO to avoid FutureWarning
                from io import StringIO
                stats_html = StringIO(str(player_stats_table))
                stats_df = pd.read_html(stats_html)[0]
                
                # Extract team abbreviations directly from HTML since pandas read_html doesn't preserve data-stat attributes
                team_abbrs = []
                rows = player_stats_table.find_all('tr')
                
                for row in rows:
                    # Skip header rows
                    if row.get('class') and 'thead' in row.get('class'):
                        continue
                    
                    # Find the team abbreviation cell (data-stat="team_name_abbr")
                    team_cell = row.find('td', {'data-stat': 'team_id'}) or row.find('td', {'data-stat': 'team_name_abbr'})
                    if team_cell:
                        team_abbrs.append(team_cell.text.strip())
                    else:
                        team_abbrs.append(None)  # Add None if not found
                
                # Add team abbreviations to dataframe if we have the right number
                if len(team_abbrs) == len(stats_df):
                    stats_df['Team_Abbr'] = team_abbrs
                
                # Clean the data
                stats_df = clean_player_per36_data(stats_df, season)
                
                # Add to dictionary
                player_per36[season] = stats_df
                logging.info(f"Successfully scraped player per 36 minutes for {season} season. Found {len(stats_df)} player records.")
            else:
                # Try alternative approach - look for any table with per minute stats
                tables = soup.find_all("table")
                found = False
                
                for i, table in enumerate(tables):
                    # Check if this looks like a player stats table
                    if table.find('th', text='PTS') and table.find('th', text='Player'):
                        logging.info(f"Found alternative player per 36 minutes table (index {i})")
                        from io import StringIO
                        stats_html = StringIO(str(table))
                        stats_df = pd.read_html(stats_html)[0]
                        
                        # Extract team abbreviations directly from HTML
                        team_abbrs = []
                        rows = table.find_all('tr')
                        
                        for row in rows:
                            # Skip header rows
                            if row.get('class') and 'thead' in row.get('class'):
                                continue
                            
                            # Find the team abbreviation cell (data-stat="team_name_abbr")
                            team_cell = row.find('td', {'data-stat': 'team_id'}) or row.find('td', {'data-stat': 'team_name_abbr'})
                            if team_cell:
                                team_abbrs.append(team_cell.text.strip())
                            else:
                                team_abbrs.append(None)  # Add None if not found
                        
                        # Add team abbreviations to dataframe if we have the right number
                        if len(team_abbrs) == len(stats_df):
                            stats_df['Team_Abbr'] = team_abbrs
                        
                        # Clean the data
                        stats_df = clean_player_per36_data(stats_df, season)
                        
                        # Add to dictionary
                        player_per36[season] = stats_df
                        logging.info(f"Successfully scraped player per 36 minutes for {season} season. Found {len(stats_df)} player records.")
                        found = True
                        break
                
                if not found:
                    logging.warning(f"No player per 36 minutes table found for {season} season. The website structure might have changed.")
        
        except Exception as e:
            logging.error(f"Error while scraping player per 36 minutes for {season} season: {str(e)}")
            continue
    
    return player_per36

def clean_player_per36_data(df, season):
    """
    Clean and format the player per 36 minutes dataframe.
    
    Args:
        df (pandas.DataFrame): Raw player per 36 minutes dataframe
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
    
    # Handle team column - prioritize our custom Team_Abbr if it exists
    if 'Team_Abbr' in df.columns:
        # Rename to standard column name
        df['Tm'] = df['Team_Abbr']
        df = df.drop(columns=['Team_Abbr'])
    elif 'Tm' not in df.columns and 'Team' in df.columns:
        # If we have 'Team' but not 'Tm', rename it
        df = df.rename(columns={'Team': 'Tm'})
    
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

def scrape_player_per100_possessions(start_season=2015, end_season=2025):
    """
    Scrape NBA player averages per 100 possessions from Basketball Reference website.
    
    Args:
        start_season (int): Starting season year (default: 2015)
        end_season (int): Ending season year (default: 2025)
        
    Returns:
        dict: Dictionary with player per 100 possessions dataframes for each season
    """
    player_per100 = {}
    
    for season in range(start_season, end_season + 1):
        logging.info(f"Scraping player per 100 possessions for {season}-{season+1} season...")
        
        # NBA player per 100 possessions URL
        url = f"https://www.basketball-reference.com/leagues/NBA_{season}_per_poss.html"
        
        # Make the throttled request
        response = throttled_request(url, max_retries=5)
        if not response:
            logging.warning(f"Failed to retrieve player per 100 possessions for {season} season.")
            continue
        
        # Parse HTML content
        soup = BeautifulSoup(response.content, 'html.parser')

        try:
            # Get Player Per 100 Possessions table - first try with the specific ID
            player_stats_table = soup.find("table", {'id': 'per_poss_stats'})
            
            if player_stats_table:
                # Use StringIO to avoid FutureWarning
                from io import StringIO
                stats_html = StringIO(str(player_stats_table))
                stats_df = pd.read_html(stats_html)[0]
                
                # Extract team abbreviations directly from HTML since pandas read_html doesn't preserve data-stat attributes
                team_abbrs = []
                rows = player_stats_table.find_all('tr')
                
                for row in rows:
                    # Skip header rows
                    if row.get('class') and 'thead' in row.get('class'):
                        continue
                    
                    # Find the team abbreviation cell (data-stat="team_name_abbr")
                    team_cell = row.find('td', {'data-stat': 'team_id'}) or row.find('td', {'data-stat': 'team_name_abbr'})
                    if team_cell:
                        team_abbrs.append(team_cell.text.strip())
                    else:
                        team_abbrs.append(None)  # Add None if not found
                
                # Add team abbreviations to dataframe if we have the right number
                if len(team_abbrs) == len(stats_df):
                    stats_df['Team_Abbr'] = team_abbrs
                
                # Clean the data
                stats_df = clean_player_per100_data(stats_df, season)
                
                # Add to dictionary
                player_per100[season] = stats_df
                logging.info(f"Successfully scraped player per 100 possessions for {season} season. Found {len(stats_df)} player records.")
            else:
                # Try alternative approach - look for any table with per possession stats
                tables = soup.find_all("table")
                found = False
                
                for i, table in enumerate(tables):
                    # Check if this looks like a player stats table
                    if table.find('th', text='PTS') and table.find('th', text='Player'):
                        logging.info(f"Found alternative player per 100 possessions table (index {i})")
                        from io import StringIO
                        stats_html = StringIO(str(table))
                        stats_df = pd.read_html(stats_html)[0]
                        
                        # Extract team abbreviations directly from HTML
                        team_abbrs = []
                        rows = table.find_all('tr')
                        
                        for row in rows:
                            # Skip header rows
                            if row.get('class') and 'thead' in row.get('class'):
                                continue
                            
                            # Find the team abbreviation cell (data-stat="team_name_abbr")
                            team_cell = row.find('td', {'data-stat': 'team_id'}) or row.find('td', {'data-stat': 'team_name_abbr'})
                            if team_cell:
                                team_abbrs.append(team_cell.text.strip())
                            else:
                                team_abbrs.append(None)  # Add None if not found
                        
                        # Add team abbreviations to dataframe if we have the right number
                        if len(team_abbrs) == len(stats_df):
                            stats_df['Team_Abbr'] = team_abbrs
                        
                        # Clean the data
                        stats_df = clean_player_per100_data(stats_df, season)
                        
                        # Add to dictionary
                        player_per100[season] = stats_df
                        logging.info(f"Successfully scraped player per 100 possessions for {season} season. Found {len(stats_df)} player records.")
                        found = True
                        break
                
                if not found:
                    logging.warning(f"No player per 100 possessions table found for {season} season. The website structure might have changed.")
        
        except Exception as e:
            logging.error(f"Error while scraping player per 100 possessions for {season} season: {str(e)}")
            continue
    
    return player_per100

def clean_player_per100_data(df, season):
    """
    Clean and format the player per 100 possessions dataframe.
    
    Args:
        df (pandas.DataFrame): Raw player per 100 possessions dataframe
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
    
    # Handle team column - prioritize our custom Team_Abbr if it exists
    if 'Team_Abbr' in df.columns:
        # Rename to standard column name
        df['Tm'] = df['Team_Abbr']
        df = df.drop(columns=['Team_Abbr'])
    elif 'Tm' not in df.columns and 'Team' in df.columns:
        # If we have 'Team' but not 'Tm', rename it
        df = df.rename(columns={'Team': 'Tm'})
    
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

def scrape_player_advanced_stats(start_season=2015, end_season=2025):
    """
    Scrape NBA player advanced statistics from Basketball Reference website.
    
    Args:
        start_season (int): Starting season year (default: 2015)
        end_season (int): Ending season year (default: 2025)
        
    Returns:
        dict: Dictionary with player advanced statistics dataframes for each season
    """
    player_advanced = {}
    
    for season in range(start_season, end_season + 1):
        logging.info(f"Scraping player advanced statistics for {season}-{season+1} season...")
        
        # NBA player advanced statistics URL
        url = f"https://www.basketball-reference.com/leagues/NBA_{season}_advanced.html"
        
        # Make the throttled request
        response = throttled_request(url, max_retries=5)
        if not response:
            logging.warning(f"Failed to retrieve player advanced statistics for {season} season.")
            continue
        
        # Parse HTML content
        soup = BeautifulSoup(response.content, 'html.parser')

        try:
            # Get Player Advanced Stats table - first try with the specific ID
            player_stats_table = soup.find("table", {'id': 'advanced_stats'})
            
            if player_stats_table:
                # Use StringIO to avoid FutureWarning
                from io import StringIO
                stats_html = StringIO(str(player_stats_table))
                stats_df = pd.read_html(stats_html)[0]
                
                # Extract team abbreviations directly from HTML since pandas read_html doesn't preserve data-stat attributes
                team_abbrs = []
                rows = player_stats_table.find_all('tr')
                
                for row in rows:
                    # Skip header rows
                    if row.get('class') and 'thead' in row.get('class'):
                        continue
                    
                    # Find the team abbreviation cell (data-stat="team_name_abbr")
                    team_cell = row.find('td', {'data-stat': 'team_id'}) or row.find('td', {'data-stat': 'team_name_abbr'})
                    if team_cell:
                        team_abbrs.append(team_cell.text.strip())
                    else:
                        team_abbrs.append(None)  # Add None if not found
                
                # Add team abbreviations to dataframe if we have the right number
                if len(team_abbrs) == len(stats_df):
                    stats_df['Team_Abbr'] = team_abbrs
                
                # Clean the data
                stats_df = clean_player_advanced_data(stats_df, season)
                
                # Add to dictionary
                player_advanced[season] = stats_df
                logging.info(f"Successfully scraped player advanced statistics for {season} season. Found {len(stats_df)} player records.")
            else:
                # Try alternative approach - look for any table with advanced stats
                tables = soup.find_all("table")
                found = False
                
                for i, table in enumerate(tables):
                    # Check if this looks like a player stats table
                    if table.find('th', text='PER') or table.find('th', text='WS') or table.find('th', text='VORP'):
                        logging.info(f"Found alternative player advanced statistics table (index {i})")
                        from io import StringIO
                        stats_html = StringIO(str(table))
                        stats_df = pd.read_html(stats_html)[0]
                        
                        # Extract team abbreviations directly from HTML
                        team_abbrs = []
                        rows = table.find_all('tr')
                        
                        for row in rows:
                            # Skip header rows
                            if row.get('class') and 'thead' in row.get('class'):
                                continue
                            
                            # Find the team abbreviation cell (data-stat="team_name_abbr")
                            team_cell = row.find('td', {'data-stat': 'team_id'}) or row.find('td', {'data-stat': 'team_name_abbr'})
                            if team_cell:
                                team_abbrs.append(team_cell.text.strip())
                            else:
                                team_abbrs.append(None)  # Add None if not found
                        
                        # Add team abbreviations to dataframe if we have the right number
                        if len(team_abbrs) == len(stats_df):
                            stats_df['Team_Abbr'] = team_abbrs
                        
                        # Clean the data
                        stats_df = clean_player_advanced_data(stats_df, season)
                        
                        # Add to dictionary
                        player_advanced[season] = stats_df
                        logging.info(f"Successfully scraped player advanced statistics for {season} season. Found {len(stats_df)} player records.")
                        found = True
                        break
                
                if not found:
                    logging.warning(f"No player advanced statistics table found for {season} season. The website structure might have changed.")
        
        except Exception as e:
            logging.error(f"Error while scraping player advanced statistics for {season} season: {str(e)}")
            continue
    
    return player_advanced

def clean_player_advanced_data(df, season):
    """
    Clean and format the player advanced statistics dataframe.
    
    Args:
        df (pandas.DataFrame): Raw player advanced statistics dataframe
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
    
    # Handle team column - prioritize our custom Team_Abbr if it exists
    if 'Team_Abbr' in df.columns:
        # Rename to standard column name
        df['Tm'] = df['Team_Abbr']
        df = df.drop(columns=['Team_Abbr'])
    elif 'Tm' not in df.columns and 'Team' in df.columns:
        # If we have 'Team' but not 'Tm', rename it
        df = df.rename(columns={'Team': 'Tm'})
    
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

def normalize_stats_by_usage(player_averages):
    """
    Normalize player statistics by their usage rate.
    
    Args:
        player_averages (dict): Dictionary of player averages dataframes by season
        
    Returns:
        dict: Dictionary of player averages dataframes with normalized stats by usage rate
    """
    normalized_player_averages = {}
    
    # Process each season
    for season, df in player_averages.items():
        # Create a copy of the dataframe to avoid modifying the original
        normalized_df = df.copy()
        
        # Check if 'USG%' column exists
        if 'USG%' not in normalized_df.columns:
            logging.warning(f"USG% column not found for {season} season. Skipping normalization.")
            normalized_player_averages[season] = normalized_df
            continue
        
        # Convert USG% to a decimal (e.g., 25.0 -> 0.25) for easier calculations
        # Replace zeros and NaNs with a small value to avoid division by zero
        normalized_df['USG_decimal'] = normalized_df['USG%'].fillna(0) / 100.0
        normalized_df['USG_decimal'] = normalized_df['USG_decimal'].replace(0, 0.001)  # Minimum usage rate to avoid division by zero
        
        # Define the stats to normalize (basic box score stats)
        stats_to_normalize = ['PTS', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'FG', 'FGA', '3P', '3PA', 'FT', 'FTA']
        
        # Normalize each stat by usage rate
        for stat in stats_to_normalize:
            if stat in normalized_df.columns:
                # Create a new column with the normalized stat
                # Handle NaN values in the stat column
                normalized_df[f'{stat}/USG'] = normalized_df[stat].fillna(0) / normalized_df['USG_decimal']
                logging.debug(f"Created {stat}/USG column for {season} season.")
        
        # Remove the temporary USG_decimal column
        normalized_df = normalized_df.drop(columns=['USG_decimal'])
        
        # Add to the normalized dictionary
        normalized_player_averages[season] = normalized_df
        
        logging.info(f"Normalized player statistics by usage rate for {season} season.")
    
    return normalized_player_averages

def enhance_player_averages_with_advanced(player_averages, player_advanced):
    """
    Enhance player averages dataframe with advanced statistics.
    
    Args:
        player_averages (dict): Dictionary of player averages dataframes by season
        player_advanced (dict): Dictionary of player advanced statistics dataframes by season
        
    Returns:
        dict: Dictionary of enhanced player averages dataframes by season
    """
    enhanced_player_averages = {}
    
    # Process each season
    for season in player_averages.keys():
        if season not in player_advanced:
            logging.warning(f"No advanced statistics data available for {season} season. Skipping enhancement.")
            enhanced_player_averages[season] = player_averages[season]
            continue
        
        # Get dataframes for the current season
        avg_df = player_averages[season]
        advanced_df = player_advanced[season]
        
        # Create a copy of the averages dataframe to avoid modifying the original
        enhanced_df = avg_df.copy()
        
        # Create a mapping key for matching players (Player + Team + Season)
        enhanced_df['match_key'] = enhanced_df['Player'] + '|' + enhanced_df['Tm'] + '|' + enhanced_df['Season']
        advanced_df['match_key'] = advanced_df['Player'] + '|' + advanced_df['Tm'] + '|' + advanced_df['Season']
        
        # Identify the stats columns to add (excluding non-stat columns)
        exclude_cols = ['Rk', 'Player', 'Age', 'Tm', 'Pos', 'G', 'GS', 'MP', 
                        'Season', 'Season_Year', 'Scraped_Date', 'match_key']
        advanced_stat_cols = [col for col in advanced_df.columns if col not in exclude_cols]
        
        # Create a dictionary to store advanced statistics for each player
        advanced_stats = {}
        for _, row in advanced_df.iterrows():
            advanced_stats[row['match_key']] = row
        
        # Add advanced statistics to the enhanced dataframe
        for stat_col in advanced_stat_cols:
            enhanced_df[stat_col] = None  # Initialize the column
            
            # Fill in the advanced statistics for each player
            for i, row in enhanced_df.iterrows():
                if row['match_key'] in advanced_stats:
                    enhanced_df.at[i, stat_col] = advanced_stats[row['match_key']][stat_col]
        
        # Remove the temporary match_key column
        enhanced_df = enhanced_df.drop(columns=['match_key'])
        
        # Add to the enhanced dictionary
        enhanced_player_averages[season] = enhanced_df
        
        logging.info(f"Enhanced player averages with advanced statistics for {season} season.")
    
    return enhanced_player_averages

def enhance_player_averages_with_per36(player_averages, player_per36):
    """
    Enhance player averages dataframe with per 36 minutes statistics.
    
    Args:
        player_averages (dict): Dictionary of player averages dataframes by season
        player_per36 (dict): Dictionary of player per 36 minutes dataframes by season
        
    Returns:
        dict: Dictionary of enhanced player averages dataframes by season
    """
    enhanced_player_averages = {}
    
    # Process each season
    for season in player_averages.keys():
        if season not in player_per36:
            logging.warning(f"No per 36 minutes data available for {season} season. Skipping enhancement.")
            enhanced_player_averages[season] = player_averages[season]
            continue
        
        # Get dataframes for the current season
        avg_df = player_averages[season]
        per36_df = player_per36[season]
        
        # Create a copy of the averages dataframe to avoid modifying the original
        enhanced_df = avg_df.copy()
        
        # Create a mapping key for matching players (Player + Team + Season)
        enhanced_df['match_key'] = enhanced_df['Player'] + '|' + enhanced_df['Tm'] + '|' + enhanced_df['Season']
        per36_df['match_key'] = per36_df['Player'] + '|' + per36_df['Tm'] + '|' + per36_df['Season']
        
        # Identify the stats columns to add (excluding non-stat columns)
        exclude_cols = ['Rk', 'Player', 'Age', 'Tm', 'Pos', 'G', 'GS', 'MP', 
                        'Season', 'Season_Year', 'Scraped_Date', 'match_key']
        per36_stat_cols = [col for col in per36_df.columns if col not in exclude_cols]
        
        # Create a dictionary to store per 36 minutes stats for each player
        per36_stats = {}
        for _, row in per36_df.iterrows():
            per36_stats[row['match_key']] = row
        
        # Add per 36 minutes stats to the enhanced dataframe
        for stat_col in per36_stat_cols:
            per36_col_name = f"{stat_col}/36 mins"
            enhanced_df[per36_col_name] = None  # Initialize the column
            
            # Fill in the per 36 minutes stats for each player
            for i, row in enhanced_df.iterrows():
                if row['match_key'] in per36_stats:
                    enhanced_df.at[i, per36_col_name] = per36_stats[row['match_key']][stat_col]
        
        # Remove the temporary match_key column
        enhanced_df = enhanced_df.drop(columns=['match_key'])
        
        # Add to the enhanced dictionary
        enhanced_player_averages[season] = enhanced_df
        
        logging.info(f"Enhanced player averages with per 36 minutes stats for {season} season.")
    
    return enhanced_player_averages

def enhance_player_averages_with_per100(player_averages, player_per100):
    """
    Enhance player averages dataframe with per 100 possessions statistics.
    
    Args:
        player_averages (dict): Dictionary of player averages dataframes by season
        player_per100 (dict): Dictionary of player per 100 possessions dataframes by season
        
    Returns:
        dict: Dictionary of enhanced player averages dataframes by season
    """
    enhanced_player_averages = {}
    
    # Process each season
    for season in player_averages.keys():
        if season not in player_per100:
            logging.warning(f"No per 100 possessions data available for {season} season. Skipping enhancement.")
            enhanced_player_averages[season] = player_averages[season]
            continue
        
        # Get dataframes for the current season
        avg_df = player_averages[season]
        per100_df = player_per100[season]
        
        # Create a copy of the averages dataframe to avoid modifying the original
        enhanced_df = avg_df.copy()
        
        # Create a mapping key for matching players (Player + Team + Season)
        enhanced_df['match_key'] = enhanced_df['Player'] + '|' + enhanced_df['Tm'] + '|' + enhanced_df['Season']
        per100_df['match_key'] = per100_df['Player'] + '|' + per100_df['Tm'] + '|' + per100_df['Season']
        
        # Identify the stats columns to add (excluding non-stat columns)
        exclude_cols = ['Rk', 'Player', 'Age', 'Tm', 'Pos', 'G', 'GS', 'MP', 
                        'Season', 'Season_Year', 'Scraped_Date', 'match_key']
        per100_stat_cols = [col for col in per100_df.columns if col not in exclude_cols]
        
        # Create a dictionary to store per 100 possessions stats for each player
        per100_stats = {}
        for _, row in per100_df.iterrows():
            per100_stats[row['match_key']] = row
        
        # Add per 100 possessions stats to the enhanced dataframe
        for stat_col in per100_stat_cols:
            per100_col_name = f"{stat_col}/100 poss"
            enhanced_df[per100_col_name] = None  # Initialize the column
            
            # Fill in the per 100 possessions stats for each player
            for i, row in enhanced_df.iterrows():
                if row['match_key'] in per100_stats:
                    enhanced_df.at[i, per100_col_name] = per100_stats[row['match_key']][stat_col]
        
        # Remove the temporary match_key column
        enhanced_df = enhanced_df.drop(columns=['match_key'])
        
        # Add to the enhanced dictionary
        enhanced_player_averages[season] = enhanced_df
        
        logging.info(f"Enhanced player averages with per 100 possessions stats for {season} season.")
    
    return enhanced_player_averages

def enhance_player_averages_with_all_stats(player_averages, player_per36, player_per100, player_advanced):
    """
    Enhance player averages dataframe with per 36 minutes, per 100 possessions, 
    and advanced statistics.
    
    Args:
        player_averages (dict): Dictionary of player averages dataframes by season
        player_per36 (dict): Dictionary of player per 36 minutes dataframes by season
        player_per100 (dict): Dictionary of player per 100 possessions dataframes by season
        player_advanced (dict): Dictionary of player advanced statistics dataframes by season
        
    Returns:
        dict: Dictionary of enhanced player averages dataframes by season
    """
    # First enhance with per 36 minutes stats
    enhanced_with_per36 = enhance_player_averages_with_per36(player_averages, player_per36)
    
    # Then enhance with per 100 possessions stats
    enhanced_with_per100 = enhance_player_averages_with_per100(enhanced_with_per36, player_per100)
    
    # Then enhance with advanced stats
    enhanced_with_advanced = enhance_player_averages_with_advanced(enhanced_with_per100, player_advanced)
    
    # Finally normalize stats by usage rate
    fully_enhanced = normalize_stats_by_usage(enhanced_with_advanced)
    
    return fully_enhanced

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
    try:
        # Set up argument parser
        parser = argparse.ArgumentParser(description='Scrape NBA data from Basketball Reference')
        parser.add_argument('--output-dir', type=str, default='./data', help='Output directory for data files')
        parser.add_argument('--start-season', type=int, default=2023, help='Starting season year (e.g., 2023 for 2023-24 season)')
        parser.add_argument('--end-season', type=int, default=2024, help='Ending season year (e.g., 2024 for 2024-25 season)')
        parser.add_argument('--standings', action='store_true', help='Scrape standings data')
        parser.add_argument('--player-stats', action='store_true', help='Scrape player statistics')
        parser.add_argument('--all', action='store_true', help='Scrape all available data')
        
        args = parser.parse_args()
        
        # Create output directory if it doesn't exist
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, 'standings'), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, 'player_stats'), exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(args.output_dir, 'scrape_log.txt')),
                logging.StreamHandler()
            ]
        )
        
        # Log script start
        logging.info("NBA Data Scraping Script Started")
        logging.info(f"Output Directory: {args.output_dir}")
        logging.info(f"Seasons: {args.start_season}-{args.end_season}")
        
        # Scrape standings if requested
        if args.standings or args.all:
            logging.info("\nScraping standings data...")
            standings_data = scrape_standings(start_season=args.start_season, end_season=args.end_season)
            
            if standings_data:
                save_standings_to_csv(standings_data, args.output_dir)
                logging.info(f"Standings data saved to {os.path.join(args.output_dir, 'standings')}")
            else:
                logging.warning("No standings data was scraped.")
        
        # Scrape player statistics if requested
        if args.player_stats or args.all:
            logging.info("\nScraping player averages per game...")
            player_averages = scrape_player_averages(start_season=args.start_season, end_season=args.end_season)
            
            if not player_averages:
                logging.error("Failed to scrape player averages. Exiting.")
                sys.exit(1)
            
            logging.info("\nScraping player per 36 minutes...")
            player_per36 = scrape_player_per36_minutes(start_season=args.start_season, end_season=args.end_season)
            
            logging.info("\nScraping player per 100 possessions...")
            player_per100 = scrape_player_per100_possessions(start_season=args.start_season, end_season=args.end_season)
            
            logging.info("\nScraping player advanced statistics...")
            player_advanced = scrape_player_advanced_stats(start_season=args.start_season, end_season=args.end_season)
            
            if player_per36 and player_per100 and player_advanced:
                # Enhance player averages with all stats
                logging.info("\nEnhancing player averages with per 36 minutes, per 100 possessions, and advanced statistics...")
                enhanced_player_averages = enhance_player_averages_with_all_stats(player_averages, player_per36, player_per100, player_advanced)
                
                # Save enhanced player averages to CSV
                save_player_averages_to_csv(enhanced_player_averages, args.output_dir)
            elif player_per36 and player_per100:
                # If only per 36 minutes and per 100 possessions scraping succeeded
                logging.warning("Failed to scrape player advanced statistics. Enhancing with per 36 minutes and per 100 possessions only.")
                enhanced_player_averages = enhance_player_averages_with_per100(enhance_player_averages_with_per36(player_averages, player_per36), player_per100)
                save_player_averages_to_csv(enhanced_player_averages, args.output_dir)
            elif player_per36 and player_advanced:
                # If only per 36 minutes and advanced statistics scraping succeeded
                logging.warning("Failed to scrape player per 100 possessions. Enhancing with per 36 minutes and advanced statistics only.")
                enhanced_player_averages = enhance_player_averages_with_advanced(enhance_player_averages_with_per36(player_averages, player_per36), player_advanced)
                save_player_averages_to_csv(enhanced_player_averages, args.output_dir)
            elif player_per100 and player_advanced:
                # If only per 100 possessions and advanced statistics scraping succeeded
                logging.warning("Failed to scrape player per 36 minutes. Enhancing with per 100 possessions and advanced statistics only.")
                enhanced_player_averages = enhance_player_averages_with_advanced(enhance_player_averages_with_per100(player_averages, player_per100), player_advanced)
                save_player_averages_to_csv(enhanced_player_averages, args.output_dir)
            elif player_per36:
                # If only per 36 minutes scraping succeeded
                logging.warning("Failed to scrape player per 100 possessions and advanced statistics. Enhancing with per 36 minutes only.")
                enhanced_player_averages = enhance_player_averages_with_per36(player_averages, player_per36)
                save_player_averages_to_csv(enhanced_player_averages, args.output_dir)
            elif player_per100:
                # If only per 100 possessions scraping succeeded
                logging.warning("Failed to scrape player per 36 minutes and advanced statistics. Enhancing with per 100 possessions only.")
                enhanced_player_averages = enhance_player_averages_with_per100(player_averages, player_per100)
                save_player_averages_to_csv(enhanced_player_averages, args.output_dir)
            elif player_advanced:
                # If only advanced statistics scraping succeeded
                logging.warning("Failed to scrape player per 36 minutes and per 100 possessions. Enhancing with advanced statistics only.")
                enhanced_player_averages = enhance_player_averages_with_advanced(player_averages, player_advanced)
                save_player_averages_to_csv(enhanced_player_averages, args.output_dir)
            else:
                # If all additional scraping failed
                logging.warning("Failed to scrape player per 36 minutes, per 100 possessions, and advanced statistics. Saving original player averages.")
                save_player_averages_to_csv(player_averages, args.output_dir)
            
            # Display sample of the data for the first season
            if args.start_season in player_averages:
                logging.info("\nSample of player averages data:")
                sample_df = player_averages[args.start_season].head(5)
                logging.info(f"\n{sample_df}")
        
        # Log script completion
        logging.info("\nNBA Data Scraping Script Completed Successfully")
    
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        logging.error(traceback.format_exc())
        sys.exit(1)