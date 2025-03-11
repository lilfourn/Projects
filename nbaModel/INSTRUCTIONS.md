# NBA Data Scraper - Instructions

This document provides instructions for using the NBA data scraper script (`scrapeData.py`), which collects data from Basketball Reference.

## Overview

The NBA data scraper is a Python script that can collect the following types of data:

1. **Team Standings**: Current NBA team standings and ratings
2. **Team Schedules**: Game schedules for all NBA teams
3. **Player Averages**: Per-game statistics for NBA players across multiple seasons

## Command-Line Arguments

The script supports various command-line arguments to customize its behavior:

| Argument | Description |
|----------|-------------|
| `--standings` | Scrape only team standings |
| `--schedules` | Scrape only team schedules |
| `--players` | Scrape only player averages |
| `--all` | Scrape all data types (default if no flags are specified) |
| `--start-season` | Starting season year for player averages (default: 2015) |
| `--end-season` | Ending season year for player averages (default: 2025) |
| `--output-dir` | Directory to save data (default: ./data) |
| `-h, --help` | Show help message and exit |

## Usage Examples

### Get Help

```bash
python3 scrapeData.py --help
```

### Scrape All Data (Default)

```bash
python3 scrapeData.py
```

or explicitly:

```bash
python3 scrapeData.py --all
```

### Scrape Only Team Standings

```bash
python3 scrapeData.py --standings
```

### Scrape Only Team Schedules

```bash
python3 scrapeData.py --schedules
```

### Scrape Only Player Averages

```bash
python3 scrapeData.py --players
```

### Scrape Player Averages for Specific Seasons

```bash
python3 scrapeData.py --players --start-season 2020 --end-season 2025
```

### Scrape Multiple Data Types

```bash
python3 scrapeData.py --standings --schedules
```

### Specify Custom Output Directory

```bash
python3 scrapeData.py --all --output-dir /path/to/custom/directory
```

## Output Files

The script saves data to the following locations (relative to the output directory):

1. **Team Standings**: `./data/team_ratings_YYYYMMDD.csv`
2. **Team Schedules**: `./data/TEAM_YYYYMMDD.csv` (one file per team)
3. **Player Averages**: `./data/player_stats/player_averages_YYYYMMDD.csv` (unified file with all seasons)

Where `YYYYMMDD` is the current date in the format year-month-day.

## Data Structure

### Team Standings

The team standings data includes:
- Team rankings
- Conference and division information
- Win-loss records
- Offensive and defensive ratings
- Net ratings
- Margin of victory

### Team Schedules

Each team's schedule includes:
- Game dates
- Opponent information
- Home/away status
- Game results (for completed games)
- Score information

### Player Averages

The player averages data includes:
- Player names and positions
- Games played and started
- Minutes per game
- Scoring statistics (points, field goals, free throws)
- Rebounding statistics (offensive, defensive, total)
- Assist, steal, and block statistics
- Season information

## Notes

- The script uses random delays between requests to avoid overloading the Basketball Reference servers.
- Data is automatically timestamped with the scrape date.
- For player averages, a unified CSV file is created with a 'Season' column indicating which season the stats are from.
