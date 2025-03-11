import pandas as pd  # For data manipulation and analysis
import requests     # For making HTTP requests
from bs4 import BeautifulSoup  # For HTML parsing
import time         # For adding delays
import os           # For file operations
import random       # For randomizing delays
import logging      # For better error tracking
import asyncio      # For asynchronous processing
import aiohttp      # For async HTTP requests
from io import StringIO  # For handling HTML strings
from fake_useragent import UserAgent  # For rotating user agents
from urllib.parse import urlparse  # For parsing URLs

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Create a session-wide user agent rotator
try:
    ua = UserAgent()
except:
    logging.warning("Could not initialize UserAgent, using default user agents")
    # Fallback list of user agents if fake_useragent fails
    USER_AGENTS = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36",
        "Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1",
        "Mozilla/5.0 (iPad; CPU OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36 Edg/91.0.864.59"
    ]

# Global rate limiter for each domain
class DomainRateLimiter:
    """Rate limiter that tracks and limits requests per domain"""
    
    def __init__(self):
        self.domains = {}
        self.lock = asyncio.Lock()
    
    async def acquire(self, url, requests_per_minute=20):
        """Acquire permission to make a request to the domain"""
        domain = urlparse(url).netloc
        
        async with self.lock:
            if domain not in self.domains:
                self.domains[domain] = {
                    'last_request': 0,
                    'semaphore': asyncio.Semaphore(3),  # Max 3 concurrent requests per domain
                    'request_times': []
                }
            
            domain_data = self.domains[domain]
            
            # Calculate time since last request
            now = time.time()
            time_since_last = now - domain_data['last_request']
            
            # Update request times list and remove old ones
            domain_data['request_times'] = [t for t in domain_data['request_times'] 
                                           if now - t < 60]  # Keep only last minute
            
            # Check if we're exceeding rate limit
            if len(domain_data['request_times']) >= requests_per_minute:
                # Calculate time to wait until we're under the limit
                wait_time = 60 - (now - domain_data['request_times'][0]) + 1
                logging.info(f"Rate limit reached for {domain}. Waiting {wait_time:.2f} seconds")
                await asyncio.sleep(wait_time)
            
            # Add jitter to avoid patterns (1-3 seconds)
            jitter = 1 + random.random() * 2
            
            # Ensure minimum delay between requests to same domain (3-5 seconds)
            if time_since_last < 3:
                delay = 3 - time_since_last + jitter
                logging.info(f"Adding delay of {delay:.2f}s for {domain}")
                await asyncio.sleep(delay)
            
            # Update tracking data
            domain_data['last_request'] = time.time()
            domain_data['request_times'].append(time.time())
            
            # Acquire semaphore for concurrent request limiting
            await domain_data['semaphore'].acquire()
            return domain_data['semaphore']
    
    def release(self, url):
        """Release the semaphore after request is complete"""
        domain = urlparse(url).netloc
        if domain in self.domains:
            self.domains[domain]['semaphore'].release()

# Create global rate limiter instance
domain_limiter = DomainRateLimiter()

def get_random_user_agent():
    """Get a random user agent string"""
    try:
        return ua.random
    except:
        return random.choice(USER_AGENTS)

def get_request_headers():
    """Generate request headers that mimic a real browser"""
    return {
        "User-Agent": get_random_user_agent(),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Cache-Control": "max-age=0",
        "TE": "Trailers",
        "DNT": "1"  # Do Not Track
    }

async def make_request(url, max_retries=3, initial_retry_delay=5):
    """
    Make an HTTP request with retry logic, rate limiting, and exponential backoff
    
    Args:
        url (str): URL to request
        max_retries (int): Maximum number of retry attempts
        initial_retry_delay (int): Initial delay between retries in seconds
        
    Returns:
        tuple: (success, response_or_error)
    """
    # Get semaphore from rate limiter
    semaphore = await domain_limiter.acquire(url)
    
    try:
        for attempt in range(max_retries):
            try:
                # Get fresh headers for each attempt
                headers = get_request_headers()
                
                # Make request with custom headers
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, headers=headers, timeout=30) as response:
                        if response.status == 200:
                            # Success - return content
                            content = await response.text()
                            return True, content
                        
                        elif response.status == 429:  # Too Many Requests
                            # Calculate exponential backoff with jitter
                            wait_time = initial_retry_delay * (2 ** attempt) + random.uniform(1, 5)
                            logging.warning(f"Rate limited (429) for {url}. Waiting {wait_time:.2f} seconds before retry.")
                            await asyncio.sleep(wait_time)
                        
                        elif response.status in (403, 503):  # Forbidden or Service Unavailable
                            # These often indicate anti-scraping measures
                            wait_time = initial_retry_delay * (2 ** attempt) + random.uniform(10, 20)
                            logging.warning(f"Possible scraping protection ({response.status}) for {url}. Waiting {wait_time:.2f} seconds.")
                            await asyncio.sleep(wait_time)
                        
                        else:
                            logging.error(f"Failed to retrieve data: Status code: {response.status} for {url}")
                            # For other errors, shorter retry delay
                            await asyncio.sleep(initial_retry_delay)
                
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                logging.error(f"Error during request for {url}: {str(e)}")
                # Network errors get a shorter retry delay
                await asyncio.sleep(initial_retry_delay)
        
        # If we get here, all retries failed
        return False, f"Max retries exceeded for {url}"
    
    finally:
        # Always release the semaphore
        domain_limiter.release(url)

async def scrape_nba_standings_async():
    """
    Scrape NBA standings data from Basketball Reference website asynchronously.
    
    Returns:
        dict: Dictionary with dataframes for different standings tables
    """
    # NBA standings URL
    url = "https://www.basketball-reference.com/leagues/NBA_2025_ratings.html"
    
    # Make HTTP request with retry logic
    success, result = await make_request(url)
    
    if not success:
        logging.error(result)
        return None
    
    # Parse HTML content
    soup = BeautifulSoup(result, 'html.parser')

    # Dictionary to store the standings data
    standings_data = {}

    try:
        # Get Team Ratings
        team_ratings_table = soup.find("table", {'id': 'ratings'})
        if team_ratings_table:
            # Use StringIO to avoid FutureWarning
            ratings_html = StringIO(str(team_ratings_table))
            ratings_df = pd.read_html(ratings_html)[0]
            
            # Handle multi-level columns if present
            if isinstance(ratings_df.columns, pd.MultiIndex):
                ratings_df.columns = ratings_df.columns.droplevel(0)
            ratings_df = clean_standings_data(ratings_df)
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

def scrape_nba_standings():
    """
    Synchronous wrapper for scrape_nba_standings_async
    
    Returns:
        dict: Dictionary with dataframes for different standings tables
    """
    return asyncio.run(scrape_nba_standings_async())
    
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
    Save the standings data to CSV files.
    
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

    # Save each dataframe to a CSV file
    for name, df in standings_data.items():
        # Create a copy of the dataframe to avoid modifying the original
        df_to_save = df.copy()
        
        # Rename columns for better readability
        for old_col, new_col in column_mapping.items():
            if old_col in df_to_save.columns:
                df_to_save.rename(columns={old_col: new_col}, inplace=True)
        
        # Create output path with current date
        output_path = os.path.join(output_dir, f"{name}_{time.strftime('%Y%m%d')}.csv")
        
        # Save to CSV
        df_to_save.to_csv(output_path, index=False)
        logging.info(f"Saved {name} to {output_path}")

async def scrape_team_schedule_async(abbr, team_name):
    """
    Scrape schedule for a single team asynchronously
    
    Args:
        abbr (str): Team abbreviation
        team_name (str): Full team name
        
    Returns:
        tuple: (success, DataFrame or error message)
    """
    # Construct URL for team schedule
    url = f"https://www.basketball-reference.com/teams/{abbr}/2025_games.html"
    
    # Make HTTP request with retry logic
    success, result = await make_request(url)
    
    if not success:
        return False, result
    
    try:
        # Parse HTML content
        soup = BeautifulSoup(result, 'html.parser')
        
        # Find the games table
        games_table = soup.find('table', {'id': 'games'})
        
        if games_table:
            # Convert table to dataframe using StringIO to avoid FutureWarning
            games_html = StringIO(str(games_table))
            schedule_df = pd.read_html(games_html)[0]
            
            # Handle multi-level columns if present
            if isinstance(schedule_df.columns, pd.MultiIndex):
                schedule_df.columns = schedule_df.columns.droplevel(0)
            
            # Clean the dataframe
            schedule_df = clean_schedule_data(schedule_df)
            
            # Add team name and abbreviation columns
            schedule_df['Team'] = team_name
            schedule_df['TeamAbbr'] = abbr
            
            logging.info(f"Successfully scraped schedule for {team_name}")
            return True, schedule_df
        else:
            logging.warning(f"No games table found for {team_name}")
            return False, f"No games table found for {team_name}"
    
    except Exception as e:
        error_msg = f"Error scraping {team_name}: {str(e)}"
        logging.error(error_msg)
        return False, error_msg

async def scrape_team_schedules_async(team_abbrs=None, max_concurrent=3):
    """
    Scrape NBA team schedules and game stats for all teams asynchronously.
    
    Args:
        team_abbrs (dict): Dictionary of team abbreviations to team names
        max_concurrent (int): Maximum number of concurrent requests
        
    Returns:
        dict: Dictionary with team schedules dataframes
    """
    # Dictionary of team abbreviations if not provided
    if team_abbrs is None:
        team_abbrs = {
            'ATL': 'Atlanta Hawks',
            'BKN': 'Brooklyn Nets',
            'BOS': 'Boston Celtics',
            'CHA': 'Charlotte Hornets',
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
            'PHX': 'Phoenix Suns',
            'POR': 'Portland Trail Blazers',
            'SAC': 'Sacramento Kings',
            'SAS': 'San Antonio Spurs',
            'TOR': 'Toronto Raptors',
            'UTA': 'Utah Jazz',
            'WAS': 'Washington Wizards'
        }
    
    # Dictionary to store team schedules
    team_schedules = {}
    
    # Create a semaphore to limit concurrent tasks
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def scrape_with_semaphore(abbr, name):
        async with semaphore:
            return await scrape_team_schedule_async(abbr, name)
    
    # Create tasks for each team
    tasks = []
    for abbr, team_name in team_abbrs.items():
        task = asyncio.create_task(scrape_with_semaphore(abbr, team_name))
        tasks.append((abbr, task))
    
    # Wait for all tasks to complete
    for abbr, task in tasks:
        success, result = await task
        if success:
            team_schedules[abbr] = result
    
    return team_schedules

def scrape_team_schedules(team_abbrs=None, max_concurrent=3):
    """
    Synchronous wrapper for scrape_team_schedules_async
    
    Args:
        team_abbrs (dict): Dictionary of team abbreviations to team names
        max_concurrent (int): Maximum number of concurrent requests
        
    Returns:
        dict: Dictionary with team schedules dataframes
    """
    return asyncio.run(scrape_team_schedules_async(team_abbrs, max_concurrent))

def clean_schedule_data(df):
    """
    Clean and format the schedule dataframe.
    
    Args:
        df (pandas.DataFrame): Raw schedule dataframe
        
    Returns:
        pandas.DataFrame: Cleaned dataframe with formatted columns
    """
    # Remove unnamed columns
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

# Run when script is executed directly
if __name__ == "__main__":
    logging.info("Starting NBA data scraping...")
    
    # Install required packages if not already installed
    try:
        import fake_useragent
    except ImportError:
        logging.info("Installing required packages...")
        import subprocess
        subprocess.check_call(["pip3", "install", "fake-useragent", "aiohttp"])
        logging.info("Packages installed successfully.")
    
    # Scrape the standings data
    logging.info("Scraping NBA standings...")
    standings = scrape_nba_standings()
    
    # Process and save data if scraping was successful
    if standings:
        # Show sample of each table
        for name, df in standings.items():
            logging.info(f"\n{name.upper()} STANDINGS:")
            print(df.head())
        
        # Save to CSV files
        save_standings_to_csv(standings)
    
    # Scrape team schedules - limit to just a few teams for testing
    logging.info("\nScraping team schedules...")
    
    # For testing, just scrape 3 teams to avoid rate limiting
    test_teams = {
        'BOS': 'Boston Celtics', 
        'LAL': 'Los Angeles Lakers', 
        'GSW': 'Golden State Warriors'
    }
    
    # Scrape the test teams with a low concurrency limit
    schedules = scrape_team_schedules(test_teams, max_concurrent=1)
    
    # Process and save schedule data if scraping was successful
    if schedules:
        # Show sample of first team's schedule
        first_team = next(iter(schedules))
        logging.info(f"\n{first_team} SCHEDULE:")
        print(schedules[first_team].head())
        
        # Save to CSV files
        save_standings_to_csv(schedules)
    else:
        logging.error("Failed to scrape team schedules.")
        
    logging.info("NBA data scraping completed.")