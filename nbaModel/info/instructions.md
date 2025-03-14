# NBA Data Scraper - Usage Instructions

This document provides instructions on how to use the NBA data scraping tool (`scrapeData.py`). The script scrapes NBA data from Basketball Reference, including team standings, player statistics, and more.

## Prerequisites

- Python 3.x
- Required packages: pandas, requests, beautifulsoup4, etc. (see requirements.txt)

## Basic Usage

The script can be run from the command line with various flags to customize the data scraping process:

```bash
python3 scrapeData.py [flags]
```

## Available Command-Line Flags

### Core Parameters

| Flag | Description | Default | Example |
|------|-------------|---------|---------|
| `--output-dir` | Directory where scraped data will be saved | `./data` | `--output-dir=./nba_data` |
| `--start-season` | Starting season year (e.g., 2023 for 2023-24 season) | 2023 | `--start-season=2020` |
| `--end-season` | Ending season year (e.g., 2024 for 2024-25 season) | 2024 | `--end-season=2023` |

### Data Selection Flags

| Flag | Description | Default | Example |
|------|-------------|---------|---------|
| `--standings` | Scrape NBA standings data | False | `--standings` |
| `--player-stats` | Scrape player statistics (includes per game, per 36 min, per 100 possessions, and advanced stats) | False | `--player-stats` |
| `--all` | Scrape all available data (standings and player stats) | False | `--all` |

## Examples

### Scrape Current Season Standings

To scrape only the standings data for the current season:

```bash
python3 scrapeData.py --standings
```

### Scrape Player Statistics

To scrape player statistics for the current season:

```bash
python3 scrapeData.py --player-stats
```

### Scrape All Data for Multiple Seasons

To scrape all data (standings and player statistics) for seasons from 2020-21 to 2023-24:

```bash
python3 scrapeData.py --all --start-season=2020 --end-season=2023
```

### Custom Output Directory

To save the scraped data to a custom directory:

```bash
python3 scrapeData.py --all --output-dir=./my_nba_data
```

## Output Files

The script creates the following directory structure and files:

```
[output-dir]/
├── standings/
│   └── team_ratings_[date].csv
├── player_stats/
│   └── player_averages_[date].csv
└── scrape_log.txt
```

- `team_ratings_[date].csv`: Contains team standings and ratings data
- `player_averages_[date].csv`: Contains comprehensive player statistics
- `scrape_log.txt`: Log file with details about the scraping process

## Data Details

### Standings Data

The standings data includes:
- Team names
- Win-loss records
- Offensive and defensive ratings
- Net ratings
- Margin of victory
- Adjusted ratings

### Player Statistics

The player statistics include:
- Per game averages (points, rebounds, assists, etc.)
- Per 36 minutes statistics
- Per 100 possessions statistics
- Advanced statistics (PER, TS%, Usage%, etc.)
- Usage-normalized statistics

## Notes

- The script implements throttling and random delays to avoid being blocked by the website
- User agent rotation is used to mimic different browsers
- Error handling with exponential backoff for retries
- Data is automatically cleaned and formatted for analysis

## Troubleshooting

If you encounter any issues:
1. Check the `scrape_log.txt` file for error messages
2. Ensure you have a stable internet connection
3. Try running with fewer data types (e.g., only `--standings` instead of `--all`)
4. For persistent issues, try with a smaller date range
