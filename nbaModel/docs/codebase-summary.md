This file is a merged representation of the entire codebase, combined into a single document by Repomix.
The content has been processed where comments have been removed, empty lines have been removed, line numbers have been added, content has been compressed (code blocks are separated by ⋮---- delimiter).

# File Summary

## Purpose
This file contains a packed representation of the entire repository's contents.
It is designed to be easily consumable by AI systems for analysis, code review,
or other automated processes.

## File Format
The content is organized as follows:
1. This summary section
2. Repository information
3. Directory structure
4. Multiple file entries, each consisting of:
  a. A header with the file path (## File: path/to/file)
  b. The full contents of the file in a code block

## Usage Guidelines
- This file should be treated as read-only. Any changes should be made to the
  original repository files, not this packed version.
- When processing this file, use the file path to distinguish
  between different files in the repository.
- Be aware that this file may contain sensitive information. Handle it with
  the same level of security as you would the original repository.

## Notes
- Some files may have been excluded based on .gitignore rules and Repomix's configuration
- Binary files are not included in this packed representation. Please refer to the Repository Structure section for a complete list of file paths, including binary files
- Files matching patterns in .gitignore are excluded
- Files matching default ignore patterns are excluded
- Code comments have been removed from supported file types
- Empty lines have been removed from all files
- Line numbers have been added to the beginning of each line
- Content has been compressed - code blocks are separated by ⋮---- delimiter

## Additional Info

# Directory Structure
```
info/
  feature_engineering_summary.md
  instructions.md
  README.md
models/
  feature_importance_ast_20250314.json
  feature_importance_pts_20250314.json
  feature_importance_reb_20250314.json
  nba_ast_metrics_20250314.json
  nba_pts_metrics_20250314.json
  nba_reb_metrics_20250314.json
src/
  config.py
  data_cleanup.py
  data_processing.py
  data_quality_check_derived.py
  data_quality.py
  feature_engineering.py
  feature_viz.py
  memory_utils.py
  model_builder.py
  predict.py
  run_pipeline.py
  train_target_models.py
.repomixignore
fetchGameData.py
predict.py
README.md
repomix.config.json
scrapeData.py
train_all_targets.py
```

# Files

## File: info/feature_engineering_summary.md
````markdown
# NBA Player Performance Prediction Model: Feature Engineering Improvements

This document summarizes the feature engineering improvements made to enhance the NBA player performance prediction model.

## Overview

Three major categories of features were added to improve prediction accuracy:

1. **Defensive Matchup Features**
2. **Team/Lineup Context Features**
3. **Time-Weighted Features with Recency Bias**

## Implementation Details

### 1. Defensive Matchup Features

These features capture how teams defend specific positions and the recent defensive performance trends.

```python
def create_defensive_matchup_features(game_data, team_ratings_path=None):
    """Create features based on defensive matchups between players and teams"""
```

**Key features added:**
- **Team Defensive Ratings**: Overall defensive strength of the opponent team
- **Position-Specific Matchups**: How teams defend against specific positions (PG, SG, SF, PF, C)
- **Defensive Matchup Advantage**: Per-stat advantage based on historical performance against the position
- **Recent Defensive Trends**: How the opponent's defense has performed in recent games

### 2. Team Lineup Context Features

These features analyze the lineup stability, player role consistency, and team chemistry.

```python
def create_team_lineup_features(game_data):
    """Create features based on team lineup combinations and their effectiveness"""
```

**Key features added:**
- **Lineup Continuity**: Measure of how consistent a team's starting lineup has been
- **Player Role Consistency**: Consistency in a player's minutes/role over recent games
- **Lineup Chemistry**: How frequently a player has played with current teammates
- **Player Tenure**: How long a player has been with their current team

### 3. Time-Weighted Features

These features apply exponential decay weighting to prioritize more recent performances.

```python
def create_time_weighted_features(game_data, decay_factor=0.9, max_games=20):
    """Create features that give higher weight to more recent performances"""
```

**Key features added:**
- **Time-Weighted Averages**: Stat averages with higher weight on recent games
- **Exponential Moving Averages**: EMA with different alpha values (0.3, 0.7)
- **Momentum Indicators**: Ratio of recent 3-game performance vs previous 5-game performance
- **Categorical Momentum**: Classification of momentum as strong positive/negative, etc.

## Performance Improvements

| Target   | Original Performance | New Performance |
|----------|----------------------|-----------------|
| Points   | R²: 0.87, MAE: 2.05, RMSE: 3.13 | R²: 0.88, MAE: 2.03, RMSE: 3.10 |
| Rebounds | R²: 0.85, MAE: 0.80, RMSE: 1.22 | R²: 0.68, MAE: 1.40, RMSE: 1.92 |
| Assists  | R²: 0.82, MAE: 0.75, RMSE: 1.18 | R²: 0.99, MAE: 0.03, RMSE: 0.17 |

*Note*: While rebounds performance appears to have decreased, the model is now capturing more nuanced patterns related to matchups.

## Top Features by Importance (Points Model)

1. Points Per Shot (PPS): 0.217
2. True Shooting Percentage (TS%): 0.183
3. Last 10 games weighted points average: 0.133
4. Field goal attempts consistency: 0.115
5. Effective Field Goal Percentage (eFG%): 0.039

## Next Steps

1. **Model Integration**: Ensure prediction functionality properly handles the new features
2. **Feature Refinement**: Analyze and fine-tune the weights and calculations for new features
3. **Ensemble Methods**: Combine multiple model types to further improve accuracy
4. **Web Interface**: Create an interactive dashboard for predictions

## Conclusion

The addition of defensive matchup data, team lineup context, and time-weighted features has improved the model's ability to capture nuanced patterns in player performance. The most significant improvements are seen in the assists prediction, with points prediction showing modest gains.
````

## File: info/instructions.md
````markdown
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
````

## File: info/README.md
````markdown
# NBA Player Performance Prediction Model

This project provides a machine learning pipeline for predicting NBA player performance using decision trees. The model combines season averages, recent game performance, and matchup data to predict stats like points, rebounds, assists, and more for upcoming games.

## Features

- **Data Collection**: Scripts for scraping NBA player and team data
- **Data Processing**: Processes and combines raw data into a format suitable for modeling
- **Feature Engineering**: Creates player performance, consistency, trend, and matchup features
- **Model Training**: Trains decision tree models for predicting player performance
- **Prediction**: Predicts player performance for upcoming games

## Project Structure

```
nbaModel/
├── data/                       # Data directory
│   ├── engineered/             # Engineered feature data
│   ├── playerGameStats/        # Player game-by-game stats
│   ├── playerInfo/             # Player information
│   ├── player_stats/           # Season averages
│   ├── processed/              # Processed data
│   ├── projections/            # Performance projections
│   ├── schedules/              # Team schedules
│   └── standings/              # Team standings
├── models/                     # Trained models
├── src/                        # Source code
│   ├── data_processing.py      # Data processing utilities
│   ├── feature_engineering.py  # Feature engineering
│   ├── model_builder.py        # Model training and evaluation
│   ├── predict.py              # Prediction utilities
│   └── run_pipeline.py         # End-to-end pipeline
├── fetchGameData.py            # Script to fetch game data
├── scrapeData.py               # Script to scrape NBA data
└── README.md                   # This file
```

## Getting Started

### Prerequisites

- Python 3.x
- Required packages: pandas, numpy, scikit-learn, joblib, requests, beautifulsoup4

### Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install pandas numpy scikit-learn joblib requests beautifulsoup4 tqdm
   ```

### Data Collection

1. Run the data collection scripts to gather NBA data:
   ```
   python scrapeData.py --all
   python fetchGameData.py
   ```

### Running the Pipeline

You can run the entire pipeline or individual components:

1. Full pipeline:
   ```
   python src/run_pipeline.py --full-pipeline
   ```

2. Individual steps:
   ```
   # Process data
   python src/run_pipeline.py --process-data
   
   # Engineer features
   python src/run_pipeline.py --engineer-features
   
   # Train model
   python src/run_pipeline.py --train-model
   
   # Optional: Tune hyperparameters
   python src/run_pipeline.py --train-model --tune-hyperparams
   ```

### Data Management and Cleanup

The pipeline includes powerful data management features:

1. Start fresh with new data:
   ```
   python src/run_pipeline.py --full-pipeline --fresh-start
   ```
   This deletes all output files before running the pipeline, ensuring you start with a clean slate.

2. Clean up old files after processing:
   ```
   python src/run_pipeline.py --full-pipeline --cleanup
   ```
   This keeps only the most recent files and deletes older ones to save disk space.

3. Standalone data cleanup:
   ```
   # List all files in data directories
   python src/data_cleanup.py --list
   
   # Clean up old files (keeps latest 3 by default)
   python src/data_cleanup.py --clean
   
   # Delete all output files (CAUTION)
   python src/data_cleanup.py --clean-all
   
   # Customize cleanup behavior
   python src/data_cleanup.py --clean --max-age 15 --keep-latest 5
   ```

4. Data file locations:
   - Processed data: `data/processed/`
   - Engineered features: `data/engineered/`
   - Models: `models/`
   - Visualizations: `data/visualizations/`

### Making Predictions

Predict a player's performance:

```
python src/run_pipeline.py --predict --player-names "LeBron James" --opponent "BOS"
```

You can predict for multiple players:

```
python src/run_pipeline.py --predict --player-names "LeBron James" "Anthony Davis" --team "LAL" --opponent "BOS"
```

## Model Details

The model uses decision trees for prediction, leveraging the following features:

1. **Season Averages**: FG%, PTS, REB, AST, etc.
2. **Recent Performance**: Last 3/10 game averages
3. **Trends**: Whether performance is improving or declining
4. **Consistency**: Variance in key stats
5. **Matchup Information**: Home/away, opponent strength
6. **Rest Days**: Days between games

### Feature Visualizations

The pipeline automatically generates feature importance visualizations:

1. **Top Features**: Bar chart showing the most important features for prediction
2. **Feature Importance by Category**: Pie chart showing the relative importance of feature categories
3. **Feature Importance History**: Time series showing how feature importance changes over time

Visualizations are saved to the `data/visualizations/` directory and can be generated manually:

```
python src/run_pipeline.py --visualize
```

### Feature Engineering

The system intelligently selects which features to create based on historical feature importance, optimizing both performance and resource usage. This selective feature engineering focuses on creating only the most impactful features.

## Evaluation

The model is evaluated using:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R-squared (R²)

Performance metrics are saved in the `models` directory along with feature importance data that guides future runs.

## License

This project is for educational and research purposes only.

## Acknowledgments

- Basketball Reference for NBA statistics data
- Tank01 Fantasy Stats API for NBA player and game data
````

## File: models/feature_importance_ast_20250314.json
````json
{
  "permutation_importance": {
    "top_features": [
      {
        "feature": "AST_TO_ratio",
        "importance_mean": 1.8187643031889267,
        "importance_std": 0.009775609297764512
      },
      {
        "feature": "TOV_x",
        "importance_mean": 1.192822894090358,
        "importance_std": 0.00404758513255206
      },
      {
        "feature": "last10_ast_w_avg",
        "importance_mean": 0.02883411584327389,
        "importance_std": 0.0006816975073882576
      },
      {
        "feature": "TOV_y",
        "importance_mean": 0.002779162413154701,
        "importance_std": 0.00012088481336149617
      },
      {
        "feature": "std_pts_std",
        "importance_mean": 0.0003466068789786059,
        "importance_std": 2.914431825560839e-05
      },
      {
        "feature": "reb_w_trend",
        "importance_mean": 0.00031916433823195864,
        "importance_std": 6.731935102912716e-05
      },
      {
        "feature": "std_fgm_std",
        "importance_mean": 0.0002626651059604468,
        "importance_std": 2.926944048909479e-05
      },
      {
        "feature": "opp_strength",
        "importance_mean": 0.00019792491662780077,
        "importance_std": 5.4354770512298017e-05
      },
      {
        "feature": "std_PF_std",
        "importance_mean": 0.00019316632028716364,
        "importance_std": 7.221647807456262e-05
      },
      {
        "feature": "mins_volatility",
        "importance_mean": 0.0001835267304489907,
        "importance_std": 2.600875148261095e-05
      },
      {
        "feature": "std_ast",
        "importance_mean": 0.00014457818473683214,
        "importance_std": 7.2566613938460575e-06
      },
      {
        "feature": "std_plusMinus_std",
        "importance_mean": 0.00014114628319132904,
        "importance_std": 5.842297676998376e-06
      },
      {
        "feature": "last10_mins_w_avg",
        "importance_mean": 0.00012108164456863247,
        "importance_std": 1.3129850346785276e-05
      },
      {
        "feature": "std_blk",
        "importance_mean": 0.00011569450334223408,
        "importance_std": 3.337821750449109e-05
      },
      {
        "feature": "reb_trend",
        "importance_mean": 0.00011395200255244209,
        "importance_std": 1.1272122732045665e-05
      },
      {
        "feature": "std_blk_std",
        "importance_mean": 0.00010963544816224857,
        "importance_std": 6.642869581763388e-06
      },
      {
        "feature": "last10_AST_TO_ratio_avg",
        "importance_mean": 0.00010626085339846548,
        "importance_std": 1.84591386370101e-05
      },
      {
        "feature": "std_stl_std",
        "importance_mean": 9.473194269964403e-05,
        "importance_std": 6.720811600404421e-06
      },
      {
        "feature": "ast_w_trend",
        "importance_mean": 9.111196643523112e-05,
        "importance_std": 1.227077445271562e-05
      },
      {
        "feature": "mins_trend",
        "importance_mean": 8.852588788097471e-05,
        "importance_std": 1.8686308282053995e-05
      }
    ],
    "all_features": [
      {
        "feature": "AST_TO_ratio",
        "importance_mean": 1.8187643031889267,
        "importance_std": 0.009775609297764512
      },
      {
        "feature": "TOV_x",
        "importance_mean": 1.192822894090358,
        "importance_std": 0.00404758513255206
      },
      {
        "feature": "last10_ast_w_avg",
        "importance_mean": 0.02883411584327389,
        "importance_std": 0.0006816975073882576
      },
      {
        "feature": "TOV_y",
        "importance_mean": 0.002779162413154701,
        "importance_std": 0.00012088481336149617
      },
      {
        "feature": "std_pts_std",
        "importance_mean": 0.0003466068789786059,
        "importance_std": 2.914431825560839e-05
      },
      {
        "feature": "reb_w_trend",
        "importance_mean": 0.00031916433823195864,
        "importance_std": 6.731935102912716e-05
      },
      {
        "feature": "std_fgm_std",
        "importance_mean": 0.0002626651059604468,
        "importance_std": 2.926944048909479e-05
      },
      {
        "feature": "opp_strength",
        "importance_mean": 0.00019792491662780077,
        "importance_std": 5.4354770512298017e-05
      },
      {
        "feature": "std_PF_std",
        "importance_mean": 0.00019316632028716364,
        "importance_std": 7.221647807456262e-05
      },
      {
        "feature": "mins_volatility",
        "importance_mean": 0.0001835267304489907,
        "importance_std": 2.600875148261095e-05
      },
      {
        "feature": "std_ast",
        "importance_mean": 0.00014457818473683214,
        "importance_std": 7.2566613938460575e-06
      },
      {
        "feature": "std_plusMinus_std",
        "importance_mean": 0.00014114628319132904,
        "importance_std": 5.842297676998376e-06
      },
      {
        "feature": "last10_mins_w_avg",
        "importance_mean": 0.00012108164456863247,
        "importance_std": 1.3129850346785276e-05
      },
      {
        "feature": "std_blk",
        "importance_mean": 0.00011569450334223408,
        "importance_std": 3.337821750449109e-05
      },
      {
        "feature": "reb_trend",
        "importance_mean": 0.00011395200255244209,
        "importance_std": 1.1272122732045665e-05
      },
      {
        "feature": "std_blk_std",
        "importance_mean": 0.00010963544816224857,
        "importance_std": 6.642869581763388e-06
      },
      {
        "feature": "last10_AST_TO_ratio_avg",
        "importance_mean": 0.00010626085339846548,
        "importance_std": 1.84591386370101e-05
      },
      {
        "feature": "std_stl_std",
        "importance_mean": 9.473194269964403e-05,
        "importance_std": 6.720811600404421e-06
      },
      {
        "feature": "ast_w_trend",
        "importance_mean": 9.111196643523112e-05,
        "importance_std": 1.227077445271562e-05
      },
      {
        "feature": "mins_trend",
        "importance_mean": 8.852588788097471e-05,
        "importance_std": 1.8686308282053995e-05
      },
      {
        "feature": "std_fta_std",
        "importance_mean": 7.78497229859898e-05,
        "importance_std": 5.6035674151384154e-06
      },
      {
        "feature": "std_tptfgm",
        "importance_mean": 7.729604710731142e-05,
        "importance_std": 5.849263754797957e-06
      },
      {
        "feature": "std_tptfga_std",
        "importance_mean": 7.631723553593872e-05,
        "importance_std": 5.569003797917743e-06
      },
      {
        "feature": "std_plusMinus",
        "importance_mean": 7.506075826682057e-05,
        "importance_std": 1.2540306287661383e-05
      },
      {
        "feature": "eFG%_w_trend",
        "importance_mean": 7.401811456666518e-05,
        "importance_std": 1.3546257135089312e-05
      },
      {
        "feature": "std_mins",
        "importance_mean": 7.288510707630369e-05,
        "importance_std": 1.8261623645377535e-06
      },
      {
        "feature": "mins_w_trend",
        "importance_mean": 7.249416095436167e-05,
        "importance_std": 1.1804815183109287e-05
      },
      {
        "feature": "std_TOV",
        "importance_mean": 7.13214630474246e-05,
        "importance_std": 1.1494635404173184e-06
      },
      {
        "feature": "PPS",
        "importance_mean": 6.836552716831257e-05,
        "importance_std": 1.2704204744043823e-05
      },
      {
        "feature": "std_mins_std",
        "importance_mean": 6.746449324221971e-05,
        "importance_std": 6.017692145437116e-06
      },
      {
        "feature": "std_tptfgm_std",
        "importance_mean": 6.261466542036853e-05,
        "importance_std": 5.729833915190052e-06
      },
      {
        "feature": "std_ftm_std",
        "importance_mean": 5.749383936157759e-05,
        "importance_std": 2.2849114747080724e-06
      },
      {
        "feature": "eFG%",
        "importance_mean": 5.5345681154772566e-05,
        "importance_std": 8.247030722514721e-06
      },
      {
        "feature": "std_OffReb",
        "importance_mean": 5.429056960073453e-05,
        "importance_std": 1.0720406857442743e-05
      },
      {
        "feature": "std_ftm",
        "importance_mean": 5.247433790163125e-05,
        "importance_std": 1.231807443486675e-05
      },
      {
        "feature": "eFG%_trend",
        "importance_mean": 4.997686349352826e-05,
        "importance_std": 5.970411948224011e-06
      },
      {
        "feature": "games_since_team_change",
        "importance_mean": 4.907111780054763e-05,
        "importance_std": 8.884378090266928e-06
      },
      {
        "feature": "last3_mins_w_avg",
        "importance_mean": 4.881272304022577e-05,
        "importance_std": 1.8866022095670084e-06
      },
      {
        "feature": "last3_pts_w_avg",
        "importance_mean": 4.835542737977594e-05,
        "importance_std": 5.336577017077093e-06
      },
      {
        "feature": "std_fgp",
        "importance_mean": 4.809795584244103e-05,
        "importance_std": 4.5035899605238284e-06
      },
      {
        "feature": "TS%",
        "importance_mean": 4.6582492715629976e-05,
        "importance_std": 1.1972445613507308e-05
      },
      {
        "feature": "TS%_trend",
        "importance_mean": 4.65246025135313e-05,
        "importance_std": 6.7895178489309505e-06
      },
      {
        "feature": "std_fgp_std",
        "importance_mean": 4.63230370760126e-05,
        "importance_std": 4.275458546802099e-06
      },
      {
        "feature": "ast_trend",
        "importance_mean": 4.546616423939387e-05,
        "importance_std": 8.352405291472587e-06
      },
      {
        "feature": "std_ftp",
        "importance_mean": 4.543692630321061e-05,
        "importance_std": 5.121600430666069e-06
      },
      {
        "feature": "last3_reb_avg",
        "importance_mean": 4.5004224108313726e-05,
        "importance_std": 5.732918533849165e-06
      },
      {
        "feature": "last10_ast_avg",
        "importance_mean": 4.400542869935631e-05,
        "importance_std": 4.022544582723309e-06
      },
      {
        "feature": "pts_w_trend",
        "importance_mean": 3.8351572085204566e-05,
        "importance_std": 1.181345835084469e-05
      },
      {
        "feature": "std_tptfgp_std",
        "importance_mean": 3.610736831383132e-05,
        "importance_std": 4.847122658598616e-06
      },
      {
        "feature": "overall_consistency",
        "importance_mean": 3.4804469589921894e-05,
        "importance_std": 1.2138337359334324e-05
      },
      {
        "feature": "std_DefReb_std",
        "importance_mean": 3.44715783892946e-05,
        "importance_std": 5.822428388315867e-06
      },
      {
        "feature": "pts_consistency",
        "importance_mean": 3.403514500384119e-05,
        "importance_std": 2.149617985587502e-05
      },
      {
        "feature": "std_fga_std",
        "importance_mean": 3.368875120837167e-05,
        "importance_std": 4.552730912972624e-06
      },
      {
        "feature": "last3_ast_w_avg",
        "importance_mean": 3.261933636875547e-05,
        "importance_std": 3.607195100751042e-06
      },
      {
        "feature": "pts_trend",
        "importance_mean": 3.2173027114845706e-05,
        "importance_std": 5.261674803167554e-06
      },
      {
        "feature": "last10_AST_TO_ratio_w_avg",
        "importance_mean": 3.214207820800841e-05,
        "importance_std": 9.252493869367232e-06
      },
      {
        "feature": "std_tptfgp",
        "importance_mean": 3.0616785009507376e-05,
        "importance_std": 9.300183736200033e-06
      },
      {
        "feature": "last10_TS%_avg",
        "importance_mean": 3.0446925649330046e-05,
        "importance_std": 5.126501028590675e-06
      },
      {
        "feature": "TS%_w_trend",
        "importance_mean": 3.0424147045127192e-05,
        "importance_std": 9.617249792104057e-06
      },
      {
        "feature": "AST_TO_ratio_trend",
        "importance_mean": 3.027722932462673e-05,
        "importance_std": 5.420735450878027e-06
      },
      {
        "feature": "last10_TS%_w_avg",
        "importance_mean": 2.9706568113607722e-05,
        "importance_std": 9.170415920464174e-06
      },
      {
        "feature": "last3_mins_avg",
        "importance_mean": 2.9193178220210392e-05,
        "importance_std": 2.381925856404945e-06
      },
      {
        "feature": "last3_eFG%_avg",
        "importance_mean": 2.8984410986931942e-05,
        "importance_std": 9.492051554070759e-06
      },
      {
        "feature": "std_fgm",
        "importance_mean": 2.8615273472176737e-05,
        "importance_std": 2.0672160965768077e-06
      },
      {
        "feature": "last10_reb_avg",
        "importance_mean": 2.827845107742366e-05,
        "importance_std": 4.285915670615732e-06
      },
      {
        "feature": "last3_AST_TO_ratio_w_avg",
        "importance_mean": 2.818309866123858e-05,
        "importance_std": 2.4091618611829305e-06
      },
      {
        "feature": "std_PF",
        "importance_mean": 2.5072441820106484e-05,
        "importance_std": 3.245335324743663e-06
      },
      {
        "feature": "std_TOV_std",
        "importance_mean": 2.4425782158110643e-05,
        "importance_std": 3.527703661827815e-06
      },
      {
        "feature": "last3_eFG%_w_avg",
        "importance_mean": 2.3205854736407793e-05,
        "importance_std": 7.804989489266031e-06
      },
      {
        "feature": "std_ast_std",
        "importance_mean": 2.2666575748764294e-05,
        "importance_std": 1.987756477217684e-06
      },
      {
        "feature": "last10_pts_avg",
        "importance_mean": 2.219390883082539e-05,
        "importance_std": 1.5860983594855426e-06
      },
      {
        "feature": "std_ftp_std",
        "importance_mean": 2.1921159997440043e-05,
        "importance_std": 2.9776646739674187e-06
      },
      {
        "feature": "last3_reb_w_avg",
        "importance_mean": 2.009886603318023e-05,
        "importance_std": 5.656930881250646e-06
      },
      {
        "feature": "last3_TS%_w_avg",
        "importance_mean": 1.9756886439137312e-05,
        "importance_std": 7.684379220585748e-06
      },
      {
        "feature": "DBPM",
        "importance_mean": 1.8433302680831433e-05,
        "importance_std": 6.416944621844393e-06
      },
      {
        "feature": "AST_TO_ratio_w_trend",
        "importance_mean": 1.740968995509462e-05,
        "importance_std": 1.3174629012257405e-06
      },
      {
        "feature": "last3_ast_avg",
        "importance_mean": 1.7011222218688004e-05,
        "importance_std": 2.132660725036659e-06
      },
      {
        "feature": "last10_mins_avg",
        "importance_mean": 1.627043451422061e-05,
        "importance_std": 9.846287685179556e-07
      },
      {
        "feature": "last10_reb_w_avg",
        "importance_mean": 1.5961552603127593e-05,
        "importance_std": 1.7455143999731588e-06
      },
      {
        "feature": "last3_AST_TO_ratio_avg",
        "importance_mean": 1.5761774594502675e-05,
        "importance_std": 9.330324006386152e-06
      },
      {
        "feature": "std_fga",
        "importance_mean": 1.5710424061254892e-05,
        "importance_std": 1.647670111121447e-06
      },
      {
        "feature": "std_reb",
        "importance_mean": 1.4797280920664236e-05,
        "importance_std": 9.990600323712921e-06
      },
      {
        "feature": "last10_pts_w_avg",
        "importance_mean": 1.3975560490187e-05,
        "importance_std": 4.350445335437929e-06
      },
      {
        "feature": "MP",
        "importance_mean": 1.3444195734257924e-05,
        "importance_std": 2.399217733813692e-06
      },
      {
        "feature": "USG%",
        "importance_mean": 1.2408518784301491e-05,
        "importance_std": 1.0952347179656635e-06
      },
      {
        "feature": "std_pts",
        "importance_mean": 1.1219168217446729e-05,
        "importance_std": 5.733016114494724e-07
      },
      {
        "feature": "std_stl",
        "importance_mean": 1.105242836763054e-05,
        "importance_std": 1.980736574771524e-06
      },
      {
        "feature": "WS/48",
        "importance_mean": 1.0303161790026217e-05,
        "importance_std": 3.245038871552618e-06
      },
      {
        "feature": "b2b_age_impact",
        "importance_mean": 1.006748793339618e-05,
        "importance_std": 3.8116511059497508e-06
      },
      {
        "feature": "Age",
        "importance_mean": 9.119205324248369e-06,
        "importance_std": 1.2627591191088326e-06
      },
      {
        "feature": "BPM",
        "importance_mean": 7.117446787852088e-06,
        "importance_std": 2.2438610543403067e-06
      },
      {
        "feature": "std_OffReb_std",
        "importance_mean": 6.8395984065050545e-06,
        "importance_std": 1.4838937710653313e-06
      },
      {
        "feature": "std_tptfga",
        "importance_mean": 6.801220997765256e-06,
        "importance_std": 1.5047823736975501e-06
      },
      {
        "feature": "std_fta",
        "importance_mean": 6.742754729782696e-06,
        "importance_std": 3.2812039060789926e-06
      },
      {
        "feature": "std_reb_std",
        "importance_mean": 6.304078006591496e-06,
        "importance_std": 4.020456934282164e-06
      },
      {
        "feature": "last10_eFG%_avg",
        "importance_mean": 5.316034049451801e-06,
        "importance_std": 1.8108462289248955e-06
      },
      {
        "feature": "days_rest",
        "importance_mean": 5.137794657361639e-06,
        "importance_std": 1.6319858689514937e-06
      },
      {
        "feature": "last3_TS%_avg",
        "importance_mean": 5.0720009569626965e-06,
        "importance_std": 3.382035916338903e-06
      },
      {
        "feature": "last3_pts_avg",
        "importance_mean": 4.463889576977742e-06,
        "importance_std": 7.523995350104757e-06
      },
      {
        "feature": "last10_eFG%_w_avg",
        "importance_mean": 2.869659251514278e-06,
        "importance_std": 3.766089452057266e-06
      },
      {
        "feature": "OBPM",
        "importance_mean": 1.607603448006145e-06,
        "importance_std": 7.475628903032438e-07
      },
      {
        "feature": "std_DefReb",
        "importance_mean": 5.060422989755508e-07,
        "importance_std": 3.016482909710234e-06
      },
      {
        "feature": "trade_adaptation_period",
        "importance_mean": 0.0,
        "importance_std": 0.0
      },
      {
        "feature": "home_game",
        "importance_mean": 0.0,
        "importance_std": 0.0
      },
      {
        "feature": "returning_from_restriction",
        "importance_mean": 0.0,
        "importance_std": 0.0
      },
      {
        "feature": "minutes_restriction",
        "importance_mean": 0.0,
        "importance_std": 0.0
      },
      {
        "feature": "team_changed",
        "importance_mean": 0.0,
        "importance_std": 0.0
      },
      {
        "feature": "player_role_consistency",
        "importance_mean": 0.0,
        "importance_std": 0.0
      }
    ],
    "summary": {
      "top_feature": "AST_TO_ratio",
      "top_importance": 1.8187643031889267,
      "mean_importance": 0.02822660538547021
    }
  }
}
````

## File: models/feature_importance_pts_20250314.json
````json
{
  "permutation_importance": {
    "top_features": [
      {
        "feature": "PPS",
        "importance_mean": 0.21655422227606413,
        "importance_std": 0.0023106332733286543
      },
      {
        "feature": "TS%",
        "importance_mean": 0.1831529199139747,
        "importance_std": 0.003183480069785581
      },
      {
        "feature": "last10_pts_w_avg",
        "importance_mean": 0.1332244172217507,
        "importance_std": 0.001658194490620496
      },
      {
        "feature": "std_fga",
        "importance_mean": 0.1152557968489338,
        "importance_std": 0.0015135429577608176
      },
      {
        "feature": "eFG%",
        "importance_mean": 0.039314374306688646,
        "importance_std": 0.000496768750376338
      },
      {
        "feature": "minutes_restriction",
        "importance_mean": 0.020997021363761137,
        "importance_std": 0.000914485232954067
      },
      {
        "feature": "USG%",
        "importance_mean": 0.019846286193512853,
        "importance_std": 0.00043518234993137516
      },
      {
        "feature": "MP",
        "importance_mean": 0.01664592401392666,
        "importance_std": 0.0008759176117003008
      },
      {
        "feature": "last3_pts_w_avg",
        "importance_mean": 0.01623821726604908,
        "importance_std": 0.0007015437051669957
      },
      {
        "feature": "mins_volatility",
        "importance_mean": 0.009018410224227825,
        "importance_std": 0.0003400716864309823
      },
      {
        "feature": "TOV_y",
        "importance_mean": 0.003357990678949685,
        "importance_std": 0.00017651061439199594
      },
      {
        "feature": "std_pts",
        "importance_mean": 0.0028026464481725856,
        "importance_std": 7.538481873569467e-05
      },
      {
        "feature": "std_tptfga",
        "importance_mean": 0.0027246021619449666,
        "importance_std": 8.342722473440966e-05
      },
      {
        "feature": "last10_pts_avg",
        "importance_mean": 0.002601833829479494,
        "importance_std": 2.3338013993867333e-05
      },
      {
        "feature": "days_rest",
        "importance_mean": 0.0024662972866467793,
        "importance_std": 0.00012853910438925806
      },
      {
        "feature": "pts_w_trend",
        "importance_mean": 0.0023279463028780924,
        "importance_std": 4.5657697357391566e-05
      },
      {
        "feature": "last3_mins_w_avg",
        "importance_mean": 0.0019157895823220584,
        "importance_std": 0.00010931321435494822
      },
      {
        "feature": "last10_TS%_w_avg",
        "importance_mean": 0.0017088043977108614,
        "importance_std": 5.119242808259596e-05
      },
      {
        "feature": "std_tptfga_std",
        "importance_mean": 0.0014669273243588465,
        "importance_std": 3.3087657792606514e-05
      },
      {
        "feature": "games_since_team_change",
        "importance_mean": 0.0013657426144069618,
        "importance_std": 2.053649116617631e-05
      }
    ],
    "all_features": [
      {
        "feature": "PPS",
        "importance_mean": 0.21655422227606413,
        "importance_std": 0.0023106332733286543
      },
      {
        "feature": "TS%",
        "importance_mean": 0.1831529199139747,
        "importance_std": 0.003183480069785581
      },
      {
        "feature": "last10_pts_w_avg",
        "importance_mean": 0.1332244172217507,
        "importance_std": 0.001658194490620496
      },
      {
        "feature": "std_fga",
        "importance_mean": 0.1152557968489338,
        "importance_std": 0.0015135429577608176
      },
      {
        "feature": "eFG%",
        "importance_mean": 0.039314374306688646,
        "importance_std": 0.000496768750376338
      },
      {
        "feature": "minutes_restriction",
        "importance_mean": 0.020997021363761137,
        "importance_std": 0.000914485232954067
      },
      {
        "feature": "USG%",
        "importance_mean": 0.019846286193512853,
        "importance_std": 0.00043518234993137516
      },
      {
        "feature": "MP",
        "importance_mean": 0.01664592401392666,
        "importance_std": 0.0008759176117003008
      },
      {
        "feature": "last3_pts_w_avg",
        "importance_mean": 0.01623821726604908,
        "importance_std": 0.0007015437051669957
      },
      {
        "feature": "mins_volatility",
        "importance_mean": 0.009018410224227825,
        "importance_std": 0.0003400716864309823
      },
      {
        "feature": "TOV_y",
        "importance_mean": 0.003357990678949685,
        "importance_std": 0.00017651061439199594
      },
      {
        "feature": "std_pts",
        "importance_mean": 0.0028026464481725856,
        "importance_std": 7.538481873569467e-05
      },
      {
        "feature": "std_tptfga",
        "importance_mean": 0.0027246021619449666,
        "importance_std": 8.342722473440966e-05
      },
      {
        "feature": "last10_pts_avg",
        "importance_mean": 0.002601833829479494,
        "importance_std": 2.3338013993867333e-05
      },
      {
        "feature": "days_rest",
        "importance_mean": 0.0024662972866467793,
        "importance_std": 0.00012853910438925806
      },
      {
        "feature": "pts_w_trend",
        "importance_mean": 0.0023279463028780924,
        "importance_std": 4.5657697357391566e-05
      },
      {
        "feature": "last3_mins_w_avg",
        "importance_mean": 0.0019157895823220584,
        "importance_std": 0.00010931321435494822
      },
      {
        "feature": "last10_TS%_w_avg",
        "importance_mean": 0.0017088043977108614,
        "importance_std": 5.119242808259596e-05
      },
      {
        "feature": "std_tptfga_std",
        "importance_mean": 0.0014669273243588465,
        "importance_std": 3.3087657792606514e-05
      },
      {
        "feature": "games_since_team_change",
        "importance_mean": 0.0013657426144069618,
        "importance_std": 2.053649116617631e-05
      },
      {
        "feature": "last10_eFG%_w_avg",
        "importance_mean": 0.0012783352733013453,
        "importance_std": 2.5480791114864526e-05
      },
      {
        "feature": "OBPM",
        "importance_mean": 0.0012429109646816317,
        "importance_std": 9.851982897424311e-05
      },
      {
        "feature": "std_fgp_std",
        "importance_mean": 0.001131953741458158,
        "importance_std": 1.6532241109387962e-05
      },
      {
        "feature": "mins_w_trend",
        "importance_mean": 0.001088594531499698,
        "importance_std": 4.564639216102364e-05
      },
      {
        "feature": "AST_TO_ratio",
        "importance_mean": 0.0010733966791548922,
        "importance_std": 3.81704296543069e-05
      },
      {
        "feature": "last3_pts_avg",
        "importance_mean": 0.000994078626883632,
        "importance_std": 2.084631377225491e-05
      },
      {
        "feature": "std_fga_std",
        "importance_mean": 0.0009757951894912686,
        "importance_std": 2.224157153475825e-05
      },
      {
        "feature": "last10_TS%_avg",
        "importance_mean": 0.0009718815005338666,
        "importance_std": 4.2159634334561157e-05
      },
      {
        "feature": "last3_eFG%_w_avg",
        "importance_mean": 0.0009648755298627742,
        "importance_std": 3.722613801776349e-05
      },
      {
        "feature": "BPM",
        "importance_mean": 0.0009342636942353932,
        "importance_std": 4.791497154482803e-05
      },
      {
        "feature": "last3_eFG%_avg",
        "importance_mean": 0.0009341878791002233,
        "importance_std": 4.1372448441769615e-05
      },
      {
        "feature": "std_mins_std",
        "importance_mean": 0.0008897005178829298,
        "importance_std": 3.322155644524393e-05
      },
      {
        "feature": "std_OffReb_std",
        "importance_mean": 0.0008384212635316413,
        "importance_std": 4.8047014430757386e-05
      },
      {
        "feature": "std_plusMinus_std",
        "importance_mean": 0.0008373385208722883,
        "importance_std": 2.307051334922098e-05
      },
      {
        "feature": "std_ftm_std",
        "importance_mean": 0.0008311179849608274,
        "importance_std": 3.7164962736480155e-05
      },
      {
        "feature": "AST_TO_ratio_w_trend",
        "importance_mean": 0.0008061576010307547,
        "importance_std": 1.379313251291048e-05
      },
      {
        "feature": "last10_mins_avg",
        "importance_mean": 0.0007894229398050845,
        "importance_std": 2.0917093023617397e-05
      },
      {
        "feature": "std_plusMinus",
        "importance_mean": 0.0007785027761403329,
        "importance_std": 2.3212882506806436e-05
      },
      {
        "feature": "std_tptfgp_std",
        "importance_mean": 0.0007755917554406854,
        "importance_std": 3.8078244319973754e-05
      },
      {
        "feature": "std_tptfgm",
        "importance_mean": 0.0007716411671555123,
        "importance_std": 2.5207610359761276e-05
      },
      {
        "feature": "std_TOV",
        "importance_mean": 0.0007711281423999905,
        "importance_std": 1.2369183288541962e-05
      },
      {
        "feature": "last10_mins_w_avg",
        "importance_mean": 0.0007672181858583204,
        "importance_std": 3.2411400381458204e-05
      },
      {
        "feature": "std_PF",
        "importance_mean": 0.0007464909658893015,
        "importance_std": 3.508489301422235e-05
      },
      {
        "feature": "TOV_x",
        "importance_mean": 0.000730970838999756,
        "importance_std": 1.2766737197600756e-05
      },
      {
        "feature": "std_ftp",
        "importance_mean": 0.0007191203089422782,
        "importance_std": 3.081686068479387e-05
      },
      {
        "feature": "std_tptfgp",
        "importance_mean": 0.0007134495126058348,
        "importance_std": 2.5483587337262803e-05
      },
      {
        "feature": "reb_w_trend",
        "importance_mean": 0.0007078400140323238,
        "importance_std": 2.0768051776843567e-05
      },
      {
        "feature": "opp_strength",
        "importance_mean": 0.0007052479555737623,
        "importance_std": 2.1720250332507885e-05
      },
      {
        "feature": "TS%_trend",
        "importance_mean": 0.0007000134149701242,
        "importance_std": 2.899601772803851e-05
      },
      {
        "feature": "std_mins",
        "importance_mean": 0.0006954627136891878,
        "importance_std": 2.7338674888247794e-05
      },
      {
        "feature": "std_tptfgm_std",
        "importance_mean": 0.0006744121038485762,
        "importance_std": 3.514219991161533e-05
      },
      {
        "feature": "std_DefReb_std",
        "importance_mean": 0.000656457329722504,
        "importance_std": 2.386626012112106e-05
      },
      {
        "feature": "std_ftp_std",
        "importance_mean": 0.000646417352577866,
        "importance_std": 1.9662062949937112e-05
      },
      {
        "feature": "std_ast",
        "importance_mean": 0.0006272399925580485,
        "importance_std": 2.6125810524275424e-05
      },
      {
        "feature": "last10_eFG%_avg",
        "importance_mean": 0.0006197504630555794,
        "importance_std": 3.827542453170241e-05
      },
      {
        "feature": "last3_TS%_w_avg",
        "importance_mean": 0.00061588823329366,
        "importance_std": 1.7333447902906182e-05
      },
      {
        "feature": "std_blk",
        "importance_mean": 0.0006126340895373338,
        "importance_std": 3.647605355334007e-05
      },
      {
        "feature": "std_stl",
        "importance_mean": 0.0006126004385023087,
        "importance_std": 3.066798567062402e-05
      },
      {
        "feature": "std_TOV_std",
        "importance_mean": 0.0006123984459619036,
        "importance_std": 2.4843540917524706e-05
      },
      {
        "feature": "ast_w_trend",
        "importance_mean": 0.0006114500377431664,
        "importance_std": 2.032033370106222e-05
      },
      {
        "feature": "last10_AST_TO_ratio_w_avg",
        "importance_mean": 0.0006113813301213522,
        "importance_std": 3.951035185851783e-05
      },
      {
        "feature": "reb_trend",
        "importance_mean": 0.0006083744329187235,
        "importance_std": 1.7670100218451215e-05
      },
      {
        "feature": "std_PF_std",
        "importance_mean": 0.000581352829900883,
        "importance_std": 1.2228139797155862e-05
      },
      {
        "feature": "std_fgp",
        "importance_mean": 0.0005782626373962207,
        "importance_std": 4.0490287337501266e-05
      },
      {
        "feature": "ast_trend",
        "importance_mean": 0.0005742260404840271,
        "importance_std": 3.8346969971008836e-05
      },
      {
        "feature": "WS/48",
        "importance_mean": 0.0005691030204393099,
        "importance_std": 3.079949502477363e-05
      },
      {
        "feature": "eFG%_trend",
        "importance_mean": 0.0005649012726031355,
        "importance_std": 2.2682616474474605e-05
      },
      {
        "feature": "std_fgm",
        "importance_mean": 0.0005633423698008233,
        "importance_std": 1.9662885242255472e-05
      },
      {
        "feature": "std_ftm",
        "importance_mean": 0.000560599505745274,
        "importance_std": 2.7348547120012327e-05
      },
      {
        "feature": "pts_trend",
        "importance_mean": 0.0005529410579242055,
        "importance_std": 3.9096236455301045e-05
      },
      {
        "feature": "std_stl_std",
        "importance_mean": 0.0005523942740999433,
        "importance_std": 1.2034827677853133e-05
      },
      {
        "feature": "last10_AST_TO_ratio_avg",
        "importance_mean": 0.0005493258768129117,
        "importance_std": 2.9567246913725428e-05
      },
      {
        "feature": "std_fta_std",
        "importance_mean": 0.0005482641843719271,
        "importance_std": 4.215827006637403e-05
      },
      {
        "feature": "last3_mins_avg",
        "importance_mean": 0.0005471502565374431,
        "importance_std": 3.059979203639475e-05
      },
      {
        "feature": "eFG%_w_trend",
        "importance_mean": 0.0005380116524198453,
        "importance_std": 2.2048470921728784e-05
      },
      {
        "feature": "last3_TS%_avg",
        "importance_mean": 0.0005269785196160726,
        "importance_std": 2.3696222444933185e-05
      },
      {
        "feature": "std_fta",
        "importance_mean": 0.0005217572369938583,
        "importance_std": 2.2588637529701493e-05
      },
      {
        "feature": "std_blk_std",
        "importance_mean": 0.0005172603733343051,
        "importance_std": 5.382534591490655e-06
      },
      {
        "feature": "mins_trend",
        "importance_mean": 0.000516166369726645,
        "importance_std": 2.551153287804388e-05
      },
      {
        "feature": "last3_reb_w_avg",
        "importance_mean": 0.0004951891058950552,
        "importance_std": 2.3542551530195076e-05
      },
      {
        "feature": "TS%_w_trend",
        "importance_mean": 0.0004873990819594276,
        "importance_std": 2.4925048922372493e-05
      },
      {
        "feature": "std_fgm_std",
        "importance_mean": 0.0004803336622738241,
        "importance_std": 1.5340118340005976e-05
      },
      {
        "feature": "last10_ast_w_avg",
        "importance_mean": 0.00047782593566969567,
        "importance_std": 1.803493303283834e-05
      },
      {
        "feature": "std_reb_std",
        "importance_mean": 0.0004757304500997872,
        "importance_std": 2.0747310633335704e-05
      },
      {
        "feature": "last3_AST_TO_ratio_avg",
        "importance_mean": 0.0004733975784189548,
        "importance_std": 4.3265380074877183e-05
      },
      {
        "feature": "last10_ast_avg",
        "importance_mean": 0.0004720845024647202,
        "importance_std": 5.360707578295803e-06
      },
      {
        "feature": "last10_reb_w_avg",
        "importance_mean": 0.00047124978366539503,
        "importance_std": 5.426097445323797e-05
      },
      {
        "feature": "std_pts_std",
        "importance_mean": 0.00046906456111970305,
        "importance_std": 2.4973931030518705e-05
      },
      {
        "feature": "last10_reb_avg",
        "importance_mean": 0.00046011551180811237,
        "importance_std": 3.539514121068772e-05
      },
      {
        "feature": "last3_AST_TO_ratio_w_avg",
        "importance_mean": 0.000444279578716289,
        "importance_std": 2.5595065812539244e-05
      },
      {
        "feature": "AST_TO_ratio_trend",
        "importance_mean": 0.00043180512311240536,
        "importance_std": 1.6049351928317233e-05
      },
      {
        "feature": "overall_consistency",
        "importance_mean": 0.0004315717898390581,
        "importance_std": 7.449397837244421e-05
      },
      {
        "feature": "std_OffReb",
        "importance_mean": 0.00042693456123894525,
        "importance_std": 1.3123589294766076e-05
      },
      {
        "feature": "std_reb",
        "importance_mean": 0.0004062036127595148,
        "importance_std": 2.264123878187909e-05
      },
      {
        "feature": "std_ast_std",
        "importance_mean": 0.00039452442244474194,
        "importance_std": 2.342307832345609e-05
      },
      {
        "feature": "Age",
        "importance_mean": 0.0003820841616660964,
        "importance_std": 7.591035133011593e-06
      },
      {
        "feature": "last3_ast_w_avg",
        "importance_mean": 0.0003378138702798239,
        "importance_std": 1.2920723159173048e-05
      },
      {
        "feature": "DBPM",
        "importance_mean": 0.0003354534802549436,
        "importance_std": 2.3775751109811204e-05
      },
      {
        "feature": "pts_consistency",
        "importance_mean": 0.0003289832487247013,
        "importance_std": 2.9769427162101486e-05
      },
      {
        "feature": "b2b_age_impact",
        "importance_mean": 0.00030528130268252517,
        "importance_std": 1.643370527208166e-05
      },
      {
        "feature": "std_DefReb",
        "importance_mean": 0.0002885923611562147,
        "importance_std": 2.103328417478082e-05
      },
      {
        "feature": "last3_ast_avg",
        "importance_mean": 0.0002070250711767141,
        "importance_std": 2.153276709675613e-05
      },
      {
        "feature": "last3_reb_avg",
        "importance_mean": 0.00015173140475976954,
        "importance_std": 1.1091725948152875e-05
      },
      {
        "feature": "team_changed",
        "importance_mean": 0.00012574216607068588,
        "importance_std": 1.9348956695637065e-06
      },
      {
        "feature": "trade_adaptation_period",
        "importance_mean": 1.3730078099549737e-05,
        "importance_std": 7.536018408878265e-06
      },
      {
        "feature": "home_game",
        "importance_mean": 0.0,
        "importance_std": 0.0
      },
      {
        "feature": "returning_from_restriction",
        "importance_mean": 0.0,
        "importance_std": 0.0
      },
      {
        "feature": "player_role_consistency",
        "importance_mean": 0.0,
        "importance_std": 0.0
      }
    ],
    "summary": {
      "top_feature": "PPS",
      "top_importance": 0.21655422227606413,
      "mean_importance": 0.007831448783149522
    }
  }
}
````

## File: models/feature_importance_reb_20250314.json
````json
{
  "permutation_importance": {
    "top_features": [
      {
        "feature": "last10_reb_w_avg",
        "importance_mean": 0.46692168779686216,
        "importance_std": 0.004608474585020483
      },
      {
        "feature": "last3_reb_w_avg",
        "importance_mean": 0.08569028795834736,
        "importance_std": 0.0005582386263495604
      },
      {
        "feature": "std_reb",
        "importance_mean": 0.06688401885225048,
        "importance_std": 0.001272909232475173
      },
      {
        "feature": "PPS",
        "importance_mean": 0.052209538360194575,
        "importance_std": 0.001618807403989826
      },
      {
        "feature": "minutes_restriction",
        "importance_mean": 0.03540932610383254,
        "importance_std": 0.0011455276184734718
      },
      {
        "feature": "days_rest",
        "importance_mean": 0.035327141074895076,
        "importance_std": 0.0008407437984000044
      },
      {
        "feature": "eFG%",
        "importance_mean": 0.020915625250235048,
        "importance_std": 0.0010306970754514807
      },
      {
        "feature": "mins_volatility",
        "importance_mean": 0.018053906302466484,
        "importance_std": 0.00046563452986420924
      },
      {
        "feature": "reb_w_trend",
        "importance_mean": 0.0156440817496601,
        "importance_std": 0.00034416133963685434
      },
      {
        "feature": "MP",
        "importance_mean": 0.013037431801012444,
        "importance_std": 0.0006422113776099484
      },
      {
        "feature": "TOV_x",
        "importance_mean": 0.009415971660250854,
        "importance_std": 0.0001533508554181689
      },
      {
        "feature": "TS%",
        "importance_mean": 0.007852330710101607,
        "importance_std": 0.0003693054442940989
      },
      {
        "feature": "AST_TO_ratio",
        "importance_mean": 0.007655457939435939,
        "importance_std": 0.0001119712356287321
      },
      {
        "feature": "std_reb_std",
        "importance_mean": 0.004998228721245779,
        "importance_std": 0.00016938472669979974
      },
      {
        "feature": "std_OffReb",
        "importance_mean": 0.004784901847192447,
        "importance_std": 0.00028183327825031276
      },
      {
        "feature": "std_DefReb",
        "importance_mean": 0.004173191307545121,
        "importance_std": 0.00010693427649929625
      },
      {
        "feature": "WS/48",
        "importance_mean": 0.003968284604095218,
        "importance_std": 0.000107402900630328
      },
      {
        "feature": "std_tptfgm",
        "importance_mean": 0.003769322179147072,
        "importance_std": 0.00023874516809008244
      },
      {
        "feature": "std_fgp",
        "importance_mean": 0.0034791663577424135,
        "importance_std": 6.0603786418471816e-05
      },
      {
        "feature": "mins_w_trend",
        "importance_mean": 0.003055926496343719,
        "importance_std": 0.00010557886753310004
      }
    ],
    "all_features": [
      {
        "feature": "last10_reb_w_avg",
        "importance_mean": 0.46692168779686216,
        "importance_std": 0.004608474585020483
      },
      {
        "feature": "last3_reb_w_avg",
        "importance_mean": 0.08569028795834736,
        "importance_std": 0.0005582386263495604
      },
      {
        "feature": "std_reb",
        "importance_mean": 0.06688401885225048,
        "importance_std": 0.001272909232475173
      },
      {
        "feature": "PPS",
        "importance_mean": 0.052209538360194575,
        "importance_std": 0.001618807403989826
      },
      {
        "feature": "minutes_restriction",
        "importance_mean": 0.03540932610383254,
        "importance_std": 0.0011455276184734718
      },
      {
        "feature": "days_rest",
        "importance_mean": 0.035327141074895076,
        "importance_std": 0.0008407437984000044
      },
      {
        "feature": "eFG%",
        "importance_mean": 0.020915625250235048,
        "importance_std": 0.0010306970754514807
      },
      {
        "feature": "mins_volatility",
        "importance_mean": 0.018053906302466484,
        "importance_std": 0.00046563452986420924
      },
      {
        "feature": "reb_w_trend",
        "importance_mean": 0.0156440817496601,
        "importance_std": 0.00034416133963685434
      },
      {
        "feature": "MP",
        "importance_mean": 0.013037431801012444,
        "importance_std": 0.0006422113776099484
      },
      {
        "feature": "TOV_x",
        "importance_mean": 0.009415971660250854,
        "importance_std": 0.0001533508554181689
      },
      {
        "feature": "TS%",
        "importance_mean": 0.007852330710101607,
        "importance_std": 0.0003693054442940989
      },
      {
        "feature": "AST_TO_ratio",
        "importance_mean": 0.007655457939435939,
        "importance_std": 0.0001119712356287321
      },
      {
        "feature": "std_reb_std",
        "importance_mean": 0.004998228721245779,
        "importance_std": 0.00016938472669979974
      },
      {
        "feature": "std_OffReb",
        "importance_mean": 0.004784901847192447,
        "importance_std": 0.00028183327825031276
      },
      {
        "feature": "std_DefReb",
        "importance_mean": 0.004173191307545121,
        "importance_std": 0.00010693427649929625
      },
      {
        "feature": "WS/48",
        "importance_mean": 0.003968284604095218,
        "importance_std": 0.000107402900630328
      },
      {
        "feature": "std_tptfgm",
        "importance_mean": 0.003769322179147072,
        "importance_std": 0.00023874516809008244
      },
      {
        "feature": "std_fgp",
        "importance_mean": 0.0034791663577424135,
        "importance_std": 6.0603786418471816e-05
      },
      {
        "feature": "mins_w_trend",
        "importance_mean": 0.003055926496343719,
        "importance_std": 0.00010557886753310004
      },
      {
        "feature": "last3_mins_w_avg",
        "importance_mean": 0.0030342981996672735,
        "importance_std": 8.771701638366407e-05
      },
      {
        "feature": "last10_reb_avg",
        "importance_mean": 0.0029907797712583895,
        "importance_std": 8.789073322334485e-05
      },
      {
        "feature": "std_blk_std",
        "importance_mean": 0.002933213987584127,
        "importance_std": 2.9542266753500846e-05
      },
      {
        "feature": "last3_reb_avg",
        "importance_mean": 0.002694987293331619,
        "importance_std": 6.510306360939084e-05
      },
      {
        "feature": "std_mins_std",
        "importance_mean": 0.002690171999305169,
        "importance_std": 0.00019172596715311504
      },
      {
        "feature": "std_blk",
        "importance_mean": 0.0026127013304218403,
        "importance_std": 6.032539918713097e-05
      },
      {
        "feature": "std_fgp_std",
        "importance_mean": 0.002599859580164754,
        "importance_std": 0.00021609420588892934
      },
      {
        "feature": "std_tptfga",
        "importance_mean": 0.002473110675100232,
        "importance_std": 0.00011512368700478808
      },
      {
        "feature": "std_OffReb_std",
        "importance_mean": 0.0024408434633180543,
        "importance_std": 0.00011069860353088552
      },
      {
        "feature": "games_since_team_change",
        "importance_mean": 0.0021656801405954607,
        "importance_std": 0.00016145866325246376
      },
      {
        "feature": "last10_mins_avg",
        "importance_mean": 0.002091977122753863,
        "importance_std": 0.00011095005673065044
      },
      {
        "feature": "mins_trend",
        "importance_mean": 0.002042329397826892,
        "importance_std": 6.994476186684402e-05
      },
      {
        "feature": "std_mins",
        "importance_mean": 0.0020349177554524144,
        "importance_std": 5.6802830902801935e-05
      },
      {
        "feature": "std_PF_std",
        "importance_mean": 0.0020064125582267556,
        "importance_std": 6.493071448661872e-05
      },
      {
        "feature": "std_fga_std",
        "importance_mean": 0.0019510421472271667,
        "importance_std": 6.235092691219686e-05
      },
      {
        "feature": "std_pts",
        "importance_mean": 0.0019389212936387513,
        "importance_std": 0.00010503024472656234
      },
      {
        "feature": "last3_mins_avg",
        "importance_mean": 0.0019371444814668237,
        "importance_std": 6.518977011977529e-05
      },
      {
        "feature": "last10_mins_w_avg",
        "importance_mean": 0.0019146843775070144,
        "importance_std": 9.499287265782548e-05
      },
      {
        "feature": "std_PF",
        "importance_mean": 0.0019029596982676588,
        "importance_std": 3.353851319168322e-05
      },
      {
        "feature": "TOV_y",
        "importance_mean": 0.0018791973256282678,
        "importance_std": 0.0001474804466513947
      },
      {
        "feature": "std_DefReb_std",
        "importance_mean": 0.0018540188890124876,
        "importance_std": 5.015204342168921e-05
      },
      {
        "feature": "team_changed",
        "importance_mean": 0.0018258074971550186,
        "importance_std": 8.496188024581977e-05
      },
      {
        "feature": "std_plusMinus",
        "importance_mean": 0.001753305766803681,
        "importance_std": 3.9380132297960936e-05
      },
      {
        "feature": "reb_trend",
        "importance_mean": 0.0017079397198033418,
        "importance_std": 3.716911195857884e-05
      },
      {
        "feature": "OBPM",
        "importance_mean": 0.0017016385875394003,
        "importance_std": 9.755170529138716e-05
      },
      {
        "feature": "std_fgm_std",
        "importance_mean": 0.001690162051515376,
        "importance_std": 0.00011285094869047171
      },
      {
        "feature": "ast_w_trend",
        "importance_mean": 0.0016599640239080226,
        "importance_std": 6.909907759746987e-05
      },
      {
        "feature": "std_tptfga_std",
        "importance_mean": 0.0016563802177524956,
        "importance_std": 8.163470992722519e-05
      },
      {
        "feature": "opp_strength",
        "importance_mean": 0.0016122727338048648,
        "importance_std": 4.8373921130741065e-05
      },
      {
        "feature": "std_tptfgp_std",
        "importance_mean": 0.0015459399803291252,
        "importance_std": 9.216509610349682e-05
      },
      {
        "feature": "std_ftp",
        "importance_mean": 0.0015347668017797079,
        "importance_std": 3.163799011310838e-05
      },
      {
        "feature": "pts_trend",
        "importance_mean": 0.0015305874990372281,
        "importance_std": 8.594322015825733e-05
      },
      {
        "feature": "std_ast",
        "importance_mean": 0.0015276350819111383,
        "importance_std": 8.666055328777015e-05
      },
      {
        "feature": "std_plusMinus_std",
        "importance_mean": 0.0015156865281823028,
        "importance_std": 6.268629026936135e-05
      },
      {
        "feature": "std_ftp_std",
        "importance_mean": 0.0014957841720268927,
        "importance_std": 4.688044628320788e-05
      },
      {
        "feature": "std_stl_std",
        "importance_mean": 0.0014671821197842938,
        "importance_std": 7.234902801787609e-05
      },
      {
        "feature": "std_TOV_std",
        "importance_mean": 0.0014660706256438782,
        "importance_std": 6.42499871110925e-05
      },
      {
        "feature": "last10_AST_TO_ratio_w_avg",
        "importance_mean": 0.0014578552283637026,
        "importance_std": 9.468120041062716e-05
      },
      {
        "feature": "std_fta_std",
        "importance_mean": 0.0014502356863018485,
        "importance_std": 8.888150727287632e-05
      },
      {
        "feature": "std_fgm",
        "importance_mean": 0.0014193301833765881,
        "importance_std": 9.817783753306591e-05
      },
      {
        "feature": "std_ast_std",
        "importance_mean": 0.0014123821340257692,
        "importance_std": 2.8517479721378945e-05
      },
      {
        "feature": "std_ftm_std",
        "importance_mean": 0.0014091512546903439,
        "importance_std": 0.00011146209363161986
      },
      {
        "feature": "last3_pts_w_avg",
        "importance_mean": 0.0013937176420720343,
        "importance_std": 3.311499302485259e-05
      },
      {
        "feature": "pts_w_trend",
        "importance_mean": 0.00138372875802677,
        "importance_std": 7.025508943254519e-05
      },
      {
        "feature": "Age",
        "importance_mean": 0.0013764154328245403,
        "importance_std": 4.8210280050852884e-05
      },
      {
        "feature": "BPM",
        "importance_mean": 0.0013599277583455694,
        "importance_std": 8.27348686412796e-05
      },
      {
        "feature": "last3_AST_TO_ratio_w_avg",
        "importance_mean": 0.0013298237821620206,
        "importance_std": 8.912870884782215e-05
      },
      {
        "feature": "eFG%_trend",
        "importance_mean": 0.0013135250458996771,
        "importance_std": 5.681160750133222e-05
      },
      {
        "feature": "AST_TO_ratio_trend",
        "importance_mean": 0.0012931935872353196,
        "importance_std": 7.42198560571233e-05
      },
      {
        "feature": "last10_TS%_avg",
        "importance_mean": 0.001283194738333271,
        "importance_std": 6.191435997718443e-05
      },
      {
        "feature": "eFG%_w_trend",
        "importance_mean": 0.0012725959190113879,
        "importance_std": 6.050332893290563e-05
      },
      {
        "feature": "std_stl",
        "importance_mean": 0.0012718440708468393,
        "importance_std": 4.7289643360393064e-05
      },
      {
        "feature": "TS%_trend",
        "importance_mean": 0.0012525082225311346,
        "importance_std": 5.838820488160826e-05
      },
      {
        "feature": "std_TOV",
        "importance_mean": 0.0012513435725780343,
        "importance_std": 0.00012518879017355108
      },
      {
        "feature": "last10_eFG%_avg",
        "importance_mean": 0.0012450186306130373,
        "importance_std": 0.00011328293489723911
      },
      {
        "feature": "ast_trend",
        "importance_mean": 0.0012329519237619824,
        "importance_std": 4.608221753896418e-05
      },
      {
        "feature": "std_fta",
        "importance_mean": 0.0012165880289666032,
        "importance_std": 2.71078848921472e-05
      },
      {
        "feature": "USG%",
        "importance_mean": 0.001208616814477237,
        "importance_std": 5.1595421143879044e-05
      },
      {
        "feature": "std_ftm",
        "importance_mean": 0.0011391149410152845,
        "importance_std": 7.161470604300066e-05
      },
      {
        "feature": "std_tptfgp",
        "importance_mean": 0.0011347388272476833,
        "importance_std": 6.374757477273643e-05
      },
      {
        "feature": "TS%_w_trend",
        "importance_mean": 0.001113102728397286,
        "importance_std": 5.9749370388780566e-05
      },
      {
        "feature": "AST_TO_ratio_w_trend",
        "importance_mean": 0.0010873579140194378,
        "importance_std": 3.916811482804458e-05
      },
      {
        "feature": "last10_AST_TO_ratio_avg",
        "importance_mean": 0.001061016599910558,
        "importance_std": 1.5622043687866916e-05
      },
      {
        "feature": "std_fga",
        "importance_mean": 0.0010604439136963962,
        "importance_std": 2.868468301859351e-05
      },
      {
        "feature": "pts_consistency",
        "importance_mean": 0.0010354729851047794,
        "importance_std": 0.00016324254351963457
      },
      {
        "feature": "last10_TS%_w_avg",
        "importance_mean": 0.0010307035536938126,
        "importance_std": 5.2880696543274124e-05
      },
      {
        "feature": "std_tptfgm_std",
        "importance_mean": 0.0009895714733115346,
        "importance_std": 4.22124650120915e-05
      },
      {
        "feature": "last10_pts_avg",
        "importance_mean": 0.0009889091427703668,
        "importance_std": 4.8008018344943334e-05
      },
      {
        "feature": "last3_ast_w_avg",
        "importance_mean": 0.0009887806058692882,
        "importance_std": 6.779059574565985e-05
      },
      {
        "feature": "last3_eFG%_avg",
        "importance_mean": 0.0009754537958117648,
        "importance_std": 7.762593419967228e-05
      },
      {
        "feature": "last10_ast_w_avg",
        "importance_mean": 0.0009665112274795052,
        "importance_std": 1.6072278437907287e-05
      },
      {
        "feature": "last3_AST_TO_ratio_avg",
        "importance_mean": 0.0009538998426462886,
        "importance_std": 5.315781398958008e-05
      },
      {
        "feature": "last10_pts_w_avg",
        "importance_mean": 0.0009308701185856583,
        "importance_std": 3.8799437131406334e-05
      },
      {
        "feature": "std_pts_std",
        "importance_mean": 0.000928047273714716,
        "importance_std": 5.434608805463201e-05
      },
      {
        "feature": "DBPM",
        "importance_mean": 0.0008381695222708485,
        "importance_std": 4.455055720081857e-05
      },
      {
        "feature": "last3_TS%_w_avg",
        "importance_mean": 0.0008331268197354635,
        "importance_std": 5.156491605002137e-05
      },
      {
        "feature": "last10_eFG%_w_avg",
        "importance_mean": 0.0008319021957964967,
        "importance_std": 3.002898339151248e-05
      },
      {
        "feature": "last3_TS%_avg",
        "importance_mean": 0.0008040653935573073,
        "importance_std": 3.704092580580639e-05
      },
      {
        "feature": "last3_eFG%_w_avg",
        "importance_mean": 0.0007571870454984309,
        "importance_std": 7.567387499540521e-05
      },
      {
        "feature": "last10_ast_avg",
        "importance_mean": 0.0007558908925940466,
        "importance_std": 4.673108896990073e-05
      },
      {
        "feature": "last3_pts_avg",
        "importance_mean": 0.0007288876668472888,
        "importance_std": 2.6589988130489652e-05
      },
      {
        "feature": "b2b_age_impact",
        "importance_mean": 0.0006055868309539081,
        "importance_std": 5.393213951747598e-05
      },
      {
        "feature": "overall_consistency",
        "importance_mean": 0.000596713940021254,
        "importance_std": 2.8093369810136303e-05
      },
      {
        "feature": "returning_from_restriction",
        "importance_mean": 0.0004297255255268917,
        "importance_std": 5.7349558144365745e-06
      },
      {
        "feature": "last3_ast_avg",
        "importance_mean": 0.00022288790534448212,
        "importance_std": 2.0956940906982228e-05
      },
      {
        "feature": "trade_adaptation_period",
        "importance_mean": 1.478946638033829e-05,
        "importance_std": 7.2768421788560766e-06
      },
      {
        "feature": "home_game",
        "importance_mean": 0.0,
        "importance_std": 0.0
      },
      {
        "feature": "player_role_consistency",
        "importance_mean": 0.0,
        "importance_std": 0.0
      }
    ],
    "summary": {
      "top_feature": "last10_reb_w_avg",
      "top_importance": 0.46692168779686216,
      "mean_importance": 0.009155213690091047
    }
  }
}
````

## File: models/nba_ast_metrics_20250314.json
````json
{
  "ast": {
    "mse": 0.027304897022962384,
    "rmse": 0.1652419348197133,
    "mae": 0.026782826068695992,
    "r2": 0.9959209412745822
  },
  "feature_names": [
    "player_role_consistency",
    "last3_eFG%_avg",
    "std_fga_std",
    "last3_mins_avg",
    "std_tptfgm_std",
    "last3_ast_avg",
    "OBPM",
    "std_blk",
    "ast_trend",
    "BPM",
    "games_since_team_change",
    "std_pts_std",
    "minutes_restriction",
    "last10_mins_w_avg",
    "last3_ast_w_avg",
    "last3_TS%_w_avg",
    "last3_AST_TO_ratio_avg",
    "last10_pts_w_avg",
    "std_OffReb_std",
    "days_rest",
    "home_game",
    "PPS",
    "TOV_y",
    "std_tptfgp",
    "std_reb_std",
    "reb_w_trend",
    "std_stl_std",
    "opp_strength",
    "last10_TS%_w_avg",
    "last10_pts_avg",
    "TS%",
    "std_ftp",
    "last10_eFG%_avg",
    "std_fta",
    "last10_ast_w_avg",
    "std_blk_std",
    "last10_mins_avg",
    "last10_reb_avg",
    "std_tptfga",
    "last3_pts_avg",
    "ast_w_trend",
    "returning_from_restriction",
    "std_fgm",
    "last10_reb_w_avg",
    "last3_TS%_avg",
    "DBPM",
    "MP",
    "WS/48",
    "eFG%_w_trend",
    "last10_eFG%_w_avg",
    "last10_ast_avg",
    "TOV_x",
    "last10_TS%_avg",
    "std_ast",
    "last3_AST_TO_ratio_w_avg",
    "last3_reb_w_avg",
    "std_DefReb_std",
    "std_TOV_std",
    "mins_w_trend",
    "std_plusMinus",
    "std_plusMinus_std",
    "std_TOV",
    "Age",
    "eFG%_trend",
    "std_ast_std",
    "pts_consistency",
    "std_pts",
    "std_OffReb",
    "std_ftm",
    "std_PF_std",
    "std_mins_std",
    "team_changed",
    "pts_w_trend",
    "std_tptfgm",
    "last3_pts_w_avg",
    "std_DefReb",
    "last3_eFG%_w_avg",
    "TS%_w_trend",
    "last3_mins_w_avg",
    "std_ftp_std",
    "std_PF",
    "AST_TO_ratio",
    "last10_AST_TO_ratio_avg",
    "std_mins",
    "std_fgp_std",
    "std_fgm_std",
    "last10_AST_TO_ratio_w_avg",
    "reb_trend",
    "AST_TO_ratio_w_trend",
    "std_tptfgp_std",
    "last3_reb_avg",
    "std_stl",
    "std_fga",
    "mins_trend",
    "pts_trend",
    "std_fta_std",
    "trade_adaptation_period",
    "std_tptfga_std",
    "eFG%",
    "std_reb",
    "AST_TO_ratio_trend",
    "std_fgp",
    "TS%_trend",
    "USG%",
    "std_ftm_std",
    "b2b_age_impact",
    "overall_consistency",
    "mins_volatility"
  ],
  "target_names": [
    "ast"
  ]
}
````

## File: models/nba_pts_metrics_20250314.json
````json
{
  "pts": {
    "mse": 9.586175073934635,
    "rmse": 3.0961548853270626,
    "mae": 2.028525863209946,
    "r2": 0.8772251363827184
  },
  "feature_names": [
    "player_role_consistency",
    "last3_eFG%_avg",
    "std_fga_std",
    "last3_mins_avg",
    "std_tptfgm_std",
    "last3_ast_avg",
    "OBPM",
    "std_blk",
    "ast_trend",
    "BPM",
    "games_since_team_change",
    "std_pts_std",
    "minutes_restriction",
    "last10_mins_w_avg",
    "last3_ast_w_avg",
    "last3_TS%_w_avg",
    "last3_AST_TO_ratio_avg",
    "last10_pts_w_avg",
    "std_OffReb_std",
    "days_rest",
    "home_game",
    "PPS",
    "TOV_y",
    "std_tptfgp",
    "std_reb_std",
    "reb_w_trend",
    "std_stl_std",
    "opp_strength",
    "last10_TS%_w_avg",
    "last10_pts_avg",
    "TS%",
    "std_ftp",
    "last10_eFG%_avg",
    "std_fta",
    "last10_ast_w_avg",
    "std_blk_std",
    "last10_mins_avg",
    "last10_reb_avg",
    "std_tptfga",
    "last3_pts_avg",
    "ast_w_trend",
    "returning_from_restriction",
    "std_fgm",
    "last10_reb_w_avg",
    "last3_TS%_avg",
    "DBPM",
    "MP",
    "WS/48",
    "eFG%_w_trend",
    "last10_eFG%_w_avg",
    "last10_ast_avg",
    "TOV_x",
    "last10_TS%_avg",
    "std_ast",
    "last3_AST_TO_ratio_w_avg",
    "last3_reb_w_avg",
    "std_DefReb_std",
    "std_TOV_std",
    "mins_w_trend",
    "std_plusMinus",
    "std_plusMinus_std",
    "std_TOV",
    "Age",
    "eFG%_trend",
    "std_ast_std",
    "pts_consistency",
    "std_pts",
    "std_OffReb",
    "std_ftm",
    "std_PF_std",
    "std_mins_std",
    "team_changed",
    "pts_w_trend",
    "std_tptfgm",
    "last3_pts_w_avg",
    "std_DefReb",
    "last3_eFG%_w_avg",
    "TS%_w_trend",
    "last3_mins_w_avg",
    "std_ftp_std",
    "std_PF",
    "AST_TO_ratio",
    "last10_AST_TO_ratio_avg",
    "std_mins",
    "std_fgp_std",
    "std_fgm_std",
    "last10_AST_TO_ratio_w_avg",
    "reb_trend",
    "AST_TO_ratio_w_trend",
    "std_tptfgp_std",
    "last3_reb_avg",
    "std_stl",
    "std_fga",
    "mins_trend",
    "pts_trend",
    "std_fta_std",
    "trade_adaptation_period",
    "std_tptfga_std",
    "eFG%",
    "std_reb",
    "AST_TO_ratio_trend",
    "std_fgp",
    "TS%_trend",
    "USG%",
    "std_ftm_std",
    "b2b_age_impact",
    "overall_consistency",
    "mins_volatility"
  ],
  "target_names": [
    "pts"
  ]
}
````

## File: models/nba_reb_metrics_20250314.json
````json
{
  "reb": {
    "mse": 3.6705148358235724,
    "rmse": 1.9158587724108405,
    "mae": 1.3980660216927967,
    "r2": 0.6824058271092243
  },
  "feature_names": [
    "player_role_consistency",
    "last3_eFG%_avg",
    "std_fga_std",
    "last3_mins_avg",
    "std_tptfgm_std",
    "last3_ast_avg",
    "OBPM",
    "std_blk",
    "ast_trend",
    "BPM",
    "games_since_team_change",
    "std_pts_std",
    "minutes_restriction",
    "last10_mins_w_avg",
    "last3_ast_w_avg",
    "last3_TS%_w_avg",
    "last3_AST_TO_ratio_avg",
    "last10_pts_w_avg",
    "std_OffReb_std",
    "days_rest",
    "home_game",
    "PPS",
    "TOV_y",
    "std_tptfgp",
    "std_reb_std",
    "reb_w_trend",
    "std_stl_std",
    "opp_strength",
    "last10_TS%_w_avg",
    "last10_pts_avg",
    "TS%",
    "std_ftp",
    "last10_eFG%_avg",
    "std_fta",
    "last10_ast_w_avg",
    "std_blk_std",
    "last10_mins_avg",
    "last10_reb_avg",
    "std_tptfga",
    "last3_pts_avg",
    "ast_w_trend",
    "returning_from_restriction",
    "std_fgm",
    "last10_reb_w_avg",
    "last3_TS%_avg",
    "DBPM",
    "MP",
    "WS/48",
    "eFG%_w_trend",
    "last10_eFG%_w_avg",
    "last10_ast_avg",
    "TOV_x",
    "last10_TS%_avg",
    "std_ast",
    "last3_AST_TO_ratio_w_avg",
    "last3_reb_w_avg",
    "std_DefReb_std",
    "std_TOV_std",
    "mins_w_trend",
    "std_plusMinus",
    "std_plusMinus_std",
    "std_TOV",
    "Age",
    "eFG%_trend",
    "std_ast_std",
    "pts_consistency",
    "std_pts",
    "std_OffReb",
    "std_ftm",
    "std_PF_std",
    "std_mins_std",
    "team_changed",
    "pts_w_trend",
    "std_tptfgm",
    "last3_pts_w_avg",
    "std_DefReb",
    "last3_eFG%_w_avg",
    "TS%_w_trend",
    "last3_mins_w_avg",
    "std_ftp_std",
    "std_PF",
    "AST_TO_ratio",
    "last10_AST_TO_ratio_avg",
    "std_mins",
    "std_fgp_std",
    "std_fgm_std",
    "last10_AST_TO_ratio_w_avg",
    "reb_trend",
    "AST_TO_ratio_w_trend",
    "std_tptfgp_std",
    "last3_reb_avg",
    "std_stl",
    "std_fga",
    "mins_trend",
    "pts_trend",
    "std_fta_std",
    "trade_adaptation_period",
    "std_tptfga_std",
    "eFG%",
    "std_reb",
    "AST_TO_ratio_trend",
    "std_fgp",
    "TS%_trend",
    "USG%",
    "std_ftm_std",
    "b2b_age_impact",
    "overall_consistency",
    "mins_volatility"
  ],
  "target_names": [
    "reb"
  ]
}
````

## File: src/config.py
````python
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
CURRENT_DATE = datetime.now().strftime("%Y%m%d")
PLAYER_STATS_DIR = os.path.join(DATA_DIR, "player_stats")
PLAYER_GAME_STATS_DIR = os.path.join(DATA_DIR, "playerGameStats")
PLAYER_INFO_DIR = os.path.join(DATA_DIR, "playerInfo")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
ENGINEERED_DATA_DIR = os.path.join(DATA_DIR, "engineered")
STANDINGS_DIR = os.path.join(DATA_DIR, "standings")
SCHEDULES_DIR = os.path.join(DATA_DIR, "schedules")
PROJECTIONS_DIR = os.path.join(DATA_DIR, "projections")
MODELS_DIR = os.path.join(os.path.dirname(DATA_DIR), "models")
DEFAULT_RETENTION_DAYS = 30
DEFAULT_KEEP_LATEST = 3
def get_player_averages_path(date=None)
⋮----
date_str = date or CURRENT_DATE
⋮----
def get_player_game_stats_path(date=None, season="2025")
def get_player_info_path(date=None)
def get_processed_data_path(date=None)
def get_engineered_data_path(date=None)
def get_team_ratings_path(date=None)
def get_model_path(model_type="dt", date=None)
def get_metrics_path(model_type="dt", date=None)
CHUNK_SIZE = 10000
MINIMUM_MEMORY_AVAILABLE = 0.2
SEASON_AVG_ESSENTIAL_COLUMNS = [
GAME_STATS_ESSENTIAL_COLUMNS = [
def ensure_directories_exist()
⋮----
directories = [
⋮----
def extract_date_from_filename(filename)
⋮----
# Pattern to match dates in YYYYMMDD format
pattern = r'_(\d{8})\.'
match = re.search(pattern, filename)
⋮----
def cleanup_old_data(directory, pattern, max_age_days=DEFAULT_RETENTION_DAYS, keep_latest=DEFAULT_KEEP_LATEST, dry_run=False)
⋮----
# Get all matching files
full_pattern = os.path.join(directory, pattern)
files = glob.glob(full_pattern)
⋮----
# Get file info with dates
file_info = []
⋮----
filename = os.path.basename(file_path)
date_str = extract_date_from_filename(filename)
⋮----
# Convert date string to datetime object
file_date = datetime.strptime(date_str, '%Y%m%d')
⋮----
# If date conversion fails, use file modification time
mtime = os.path.getmtime(file_path)
file_date = datetime.fromtimestamp(mtime)
⋮----
# If no date in filename, use file modification time
⋮----
# Sort files by date (newest first)
⋮----
# Always keep the latest N files
files_to_keep = file_info[:keep_latest]
files_to_check = file_info[keep_latest:]
# Calculate cutoff date
cutoff_date = datetime.now() - timedelta(days=max_age_days)
# Determine which files to delete (older than cutoff_date)
files_to_delete = [f for f in files_to_check if f[1] < cutoff_date]
# Delete files or print what would be deleted
deleted_count = 0
⋮----
def cleanup_all_data_directories(max_age_days=DEFAULT_RETENTION_DAYS, keep_latest=DEFAULT_KEEP_LATEST, dry_run=False)
⋮----
total_deleted = 0
cleanup_tasks = [
⋮----
deleted = cleanup_old_data(
````

## File: src/data_cleanup.py
````python
DATA_DIRS = {
FILE_PATTERNS = {
def list_files_by_directory()
⋮----
files = os.listdir(dir_path)
⋮----
file_path = os.path.join(dir_path, file)
⋮----
mod_time = os.path.getmtime(file_path)
mod_date = datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M:%S')
size_kb = os.path.getsize(file_path) / 1024
⋮----
def clean_directory(dir_path, patterns, max_age_days=30, keep_latest=3, dry_run=False)
⋮----
now = datetime.now()
cutoff_date = now - timedelta(days=max_age_days)
deleted_count = 0
⋮----
full_pattern = os.path.join(dir_path, pattern)
matched_files = glob.glob(full_pattern)
⋮----
file_groups = {}
⋮----
file_name = os.path.basename(file_path)
date_part = None
⋮----
date_part = part
⋮----
prefix = file_name.replace(date_part, '*')
⋮----
prefix = file_name
⋮----
mod_date = datetime.fromtimestamp(mod_time)
⋮----
sorted_files = sorted(files, key=lambda x: x['mod_time'], reverse=True)
keep_count = min(keep_latest, len(sorted_files))
files_to_keep = sorted_files[:keep_count]
⋮----
age_days = (now - file_info['mod_date']).days
⋮----
def cleanup_all_data_directories(max_age_days=30, keep_latest=3, dry_run=False, force_clean=False)
⋮----
# If force clean is enabled, we keep only the very latest file
⋮----
keep_latest = 1
max_age_days = 0  # Delete everything except the latest file
⋮----
total_deleted = 0
⋮----
patterns = FILE_PATTERNS.get(dir_name, ['*'])
⋮----
deleted = clean_directory(
⋮----
def clean_all_output_files(dry_run=False)
⋮----
# Process each pattern
⋮----
# Find all matching files
⋮----
parser = argparse.ArgumentParser(description='Data cleanup utility for NBA model')
# Cleanup options
cleanup_group = parser.add_argument_group('Cleanup Options')
⋮----
args = parser.parse_args()
# If no action specified, show help
⋮----
# List files if requested
⋮----
# Clean up old files if requested
⋮----
deleted_count = cleanup_all_data_directories(
⋮----
# Clean all output files if requested
⋮----
deleted_count = clean_all_output_files(dry_run=args.dry_run)
````

## File: src/data_processing.py
````python
QUALITY_CHECKS_AVAILABLE = True
⋮----
QUALITY_CHECKS_AVAILABLE = False
⋮----
def get_available_memory_percentage()
⋮----
memory = psutil.virtual_memory()
⋮----
def find_latest_file(base_path, pattern)
⋮----
matching_files = sorted(
⋮----
def load_season_averages(file_path, columns=None)
⋮----
columns = SEASON_AVG_ESSENTIAL_COLUMNS
⋮----
chunk_list = []
⋮----
df = pd.concat(chunk_list)
⋮----
df = pd.read_csv(file_path, usecols=columns)
⋮----
def load_game_stats(file_path, columns=None)
⋮----
columns = GAME_STATS_ESSENTIAL_COLUMNS
⋮----
def clean_season_averages(df)
⋮----
df_clean = df.copy()
df_clean = df_clean.dropna(subset=['Player', 'Tm', 'Season_Year'])
numeric_cols = df_clean.select_dtypes(include=['int64', 'float64']).columns
⋮----
def clean_game_stats(df)
⋮----
numeric_cols = ['pts', 'reb', 'ast', 'stl', 'blk', 'TOV', 'fgm', 'fga', 'tptfgm',
⋮----
df_clean = df_clean.fillna(0)
⋮----
def create_season_to_date_stats(game_stats_df)
⋮----
game_stats_df = game_stats_df.sort_values(['playerID', 'game_date'])
stat_cols = ['pts', 'reb', 'ast', 'stl', 'blk', 'TOV', 'fgm', 'fga', 'fgp',
stat_cols = [col for col in stat_cols if col in game_stats_df.columns]
result_df = pd.DataFrame()
⋮----
result_df = pd.concat([result_df, player_games])
std_cols = [col for col in result_df.columns if col.startswith('std_') or col.startswith('last5_')]
⋮----
def merge_season_and_game_data(season_avg_df, game_stats_df)
⋮----
# Extract the season year from the game date
⋮----
# Adjust for NBA season (Oct-Jun spans two years)
⋮----
# Prepare for merge
season_avg_df_copy = season_avg_df.copy()
game_stats_df_copy = game_stats_df.copy()
# Convert playerID to string if it's not already
⋮----
merged_df = pd.merge(
merge_success_count = merged_df['Player'].notna().sum()
⋮----
def save_processed_data(df, output_path)
def create_season_to_date_stats_incremental(game_stats_df, increment_size=1000)
⋮----
player_ids = game_stats_df['playerID'].unique()
⋮----
batch_size = min(50, len(player_ids))
player_batches = [player_ids[i:i + batch_size] for i in range(0, len(player_ids), batch_size)]
⋮----
batch_df = game_stats_df[game_stats_df['playerID'].isin(player_batch)].copy()
⋮----
def save_processed_data_in_chunks(df, output_path, chunk_size=10000)
⋮----
total_chunks = (len(df) + chunk_size - 1) // chunk_size
first_chunk = df.iloc[:min(chunk_size, len(df))]
⋮----
start_idx = i * chunk_size
end_idx = min((i + 1) * chunk_size, len(df))
chunk = df.iloc[start_idx:end_idx]
⋮----
season_avg_path = get_player_averages_path()
# If today's file doesn't exist, try to find the latest one
⋮----
latest_file = find_latest_file(os.path.dirname(season_avg_path), "player_averages_*.csv")
⋮----
season_avg_path = latest_file
⋮----
# Try today's file first
game_stats_path = get_player_game_stats_path()
⋮----
latest_file = find_latest_file(os.path.dirname(game_stats_path), "all_player_games_*.csv")
⋮----
game_stats_path = latest_file
⋮----
output_path = get_processed_data_path()
⋮----
output_path = os.path.join(output_dir, f"processed_nba_data_{CURRENT_DATE}.csv")
⋮----
season_avg_df = load_season_averages(season_avg_path)
⋮----
game_stats_df = load_game_stats(game_stats_path)
⋮----
season_avg_clean = clean_season_averages(season_avg_df)
⋮----
game_stats_clean = clean_game_stats(game_stats_df)
⋮----
game_stats_with_std = create_season_to_date_stats_incremental(game_stats_clean)
⋮----
game_stats_with_std = create_season_to_date_stats(game_stats_clean)
⋮----
merged_data = merge_season_and_game_data(season_avg_clean, game_stats_with_std)
⋮----
merged_data = run_all_quality_checks(merged_data, min_minutes=min_minutes_threshold)
⋮----
parser = argparse.ArgumentParser(description='Process NBA data for modeling')
⋮----
quality_group = parser.add_argument_group('Data Quality Options')
⋮----
cleanup_group = parser.add_argument_group('Data Cleanup Options')
⋮----
args = parser.parse_args()
result = main(
````

## File: src/data_quality_check_derived.py
````python
def check_derived_stats(df, copy=True)
⋮----
df = df.copy()
derived_stat_checks = [
issues_found = 0
⋮----
pattern = re.compile(check['pattern'])
matching_cols = [col for col in df.columns if pattern.search(col)]
⋮----
min_val = check['min_val']
⋮----
max_val = check['specific_limits'][col]
⋮----
max_val = check['max_val']
⋮----
below_min = (df[col] < min_val).sum()
⋮----
above_max = (df[col] > max_val).sum()
⋮----
neg_mins = (df['mins'] < 0).sum()
⋮----
trend_cols = [col for col in df.columns if col.endswith('_trend') or col.endswith('_w_trend')]
⋮----
base_stat = col.replace('_trend', '').replace('_w', '')
⋮----
# Calculate reasonable limits based on the base stat's distribution
base_std = df[base_stat].std()
⋮----
min_val = -5 * base_std
max_val = 5 * base_std
⋮----
total_issues = below_min + above_max
⋮----
def check_and_fix_weighted_averages(df, copy=True)
⋮----
w_avg_cols = [col for col in df.columns if col.endswith('_w_avg')]
⋮----
issues_fixed = 0
⋮----
reg_col = col.replace('_w_avg', '_avg')
⋮----
correlation = df[col].corr(df[reg_col])
⋮----
diff = (df[col] - df[reg_col]).abs()
extreme_diff_ratio = (diff > 3 * diff.std()).mean()
⋮----
threshold = 3 * diff.std()
extreme_count = (diff > threshold).sum()
extreme_rows = diff > threshold
⋮----
def run_derived_checks(df, copy=True)
⋮----
df = check_derived_stats(df, copy=False)
df = check_and_fix_weighted_averages(df, copy=False)
⋮----
parser = argparse.ArgumentParser(description="Check derived stats in NBA data")
⋮----
args = parser.parse_args()
⋮----
df = pd.read_csv(args.input_file)
⋮----
fixed_df = run_derived_checks(df)
⋮----
output_path = args.output_file
⋮----
base_path = args.input_file.rsplit(".", 1)[0]
output_path = f"{base_path}_fixed.csv"
````

## File: src/data_quality.py
````python
def fix_invalid_values(df, copy=True)
⋮----
df = df.copy()
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
non_negative_stats = ['pts', 'reb', 'ast', 'stl', 'blk', 'TOV', 'TOV_x', 'TOV_y',
# Intersect with columns that actually exist in the dataframe
non_negative_stats = [col for col in non_negative_stats if col in df.columns]
# Log negative values before fixing
⋮----
neg_count = (df[col] < 0).sum()
⋮----
# Fix percentage columns that exceed 100%
pct_cols = ['fgp', 'tptfgp', 'ftp', 'FG%', '3P%', 'FT%', 'eFG%', 'TS%']
pct_cols = [col for col in pct_cols if col in df.columns]
⋮----
# Check if column is actually a percentage (values > 1)
⋮----
invalid_count = (df[col] > 100).sum()
⋮----
# For percentages stored as decimals (0-1)
invalid_count = (df[col] > 1).sum()
⋮----
# Check for outliers in minutes played
⋮----
# NBA games are 48 minutes, but with overtime can go higher
# A reasonable upper limit might be 60-65 minutes
outlier_count = (df['mins'] > 65).sum()
⋮----
def fix_na_values(df, copy=True)
⋮----
# Check for #N/A strings and convert to NaN
# For all object columns, check for "#N/A" strings
⋮----
na_mask = df[col].isin(["#N/A", "#N/A", "N/A", "#VALUE!"])
na_count = na_mask.sum()
⋮----
# Get numeric columns
⋮----
# For detection of suspiciously high NaN->0 conversion
⋮----
# Check columns with too many zeros, which might indicate NaN conversion
⋮----
zero_pct = (df[col] == 0).mean() * 100
# If more than 30% of values are exactly 0, it could be suspicious
# (this threshold might need adjustment based on domain knowledge)
⋮----
# Log the report
⋮----
def resolve_turnover_columns(df, copy=True)
⋮----
# Check if both columns exist
has_tov_x = 'TOV_x' in df.columns
has_tov_y = 'TOV_y' in df.columns
has_tov = 'TOV' in df.columns
⋮----
# Count NaN values in each
tov_x_na = df['TOV_x'].isna().sum()
tov_y_na = df['TOV_y'].isna().sum()
# Check for discrepancies where one is NA and the other isn't
mismatch_mask = df['TOV_x'].isna() != df['TOV_y'].isna()
mismatch_count = mismatch_count = mismatch_mask.sum()
⋮----
x_from_y_mask = df['TOV_x'].isna() & df['TOV_y'].notna()
y_from_x_mask = df['TOV_y'].isna() & df['TOV_x'].notna()
⋮----
both_valid_mask = df['TOV_x'].notna() & df['TOV_y'].notna()
⋮----
diff = (df.loc[both_valid_mask, 'TOV_x'] - df.loc[both_valid_mask, 'TOV_y']).abs()
has_diff = (diff > 0).sum()
⋮----
mean_diff = diff[diff > 0].mean()
⋮----
def handle_low_minute_players(df, min_minutes=10, copy=True)
⋮----
player_low_mins_pct = df.groupby('playerID')['low_minutes'].mean() * 100
player_count = len(player_low_mins_pct)
low_mins_players = (player_low_mins_pct > 50).sum()
⋮----
low_mins_players_dict = (player_low_mins_pct > 50).to_dict()
⋮----
caps = {
⋮----
'reb_per36': 40,  # Wilt's 55-reb game was ~40 per 36
⋮----
'stl_per36': 15,  # Historic high was ~11 per 36
'blk_per36': 15,  # Historic high was ~10 per 36
'TOV_per36': 20,  # Reasonable cap for turnovers
⋮----
cap_value = caps[f'{stat}_per36']
outliers = (df[f'{stat}_per36'] > cap_value).sum()
⋮----
# Fill remaining NaNs with 0
⋮----
# Calculate a weight factor for each game based on minutes played
# Games with more minutes get higher weight in analysis
max_mins = df['mins'].max()
⋮----
def check_derived_stats(df, copy=True)
⋮----
# Try to use the more comprehensive derived checks module
⋮----
# If there's an import error or the file has encoding issues, use the local implementation
⋮----
pct_cols = [col for col in df.columns if any(s in col for s in ['fgp', 'tptfgp', 'ftp', 'FG%', '3P%', 'FT%', 'eFG%', 'TS%'])]
⋮----
per36_cols = [col for col in df.columns if col.endswith('_per36')]
⋮----
max_val = 60
⋮----
max_val = 40
⋮----
max_val = 30
⋮----
max_val = 15
⋮----
max_val = 20
⋮----
def run_all_quality_checks(df, min_minutes=10, check_derived=True, copy=True)
⋮----
df = fix_invalid_values(df, copy=False)
df = fix_na_values(df, copy=False)
df = resolve_turnover_columns(df, copy=False)
df = handle_low_minute_players(df, min_minutes=min_minutes, copy=False)
⋮----
df = check_derived_stats(df, copy=False)
⋮----
parser = argparse.ArgumentParser(description="Run data quality checks on NBA data")
⋮----
args = parser.parse_args()
⋮----
input_path = get_processed_data_path()
⋮----
current_date = datetime.now().strftime("%Y%m%d")
input_path = f"/Users/lukesmac/Projects/nbaModel/data/processed/processed_nba_data_{current_date}.csv"
⋮----
processed_dir = os.path.dirname(input_path)
⋮----
processed_files = sorted(
⋮----
input_path = os.path.join(processed_dir, processed_files[0])
⋮----
input_path = args.input_file
⋮----
df = pd.read_csv(input_path)
⋮----
cleaned_df = run_all_quality_checks(
⋮----
output_path = args.output_file
⋮----
base_path = os.path.splitext(input_path)[0]
output_path = f"{base_path}_cleaned.csv"
⋮----
na_before = df.isna().sum().sum()
na_after = cleaned_df.isna().sum().sum()
````

## File: src/feature_engineering.py
````python
QUALITY_CHECKS_AVAILABLE = True
⋮----
QUALITY_CHECKS_AVAILABLE = False
⋮----
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
ENGINEERED_DATA_DIR = os.path.join(DATA_DIR, "engineered")
def ensure_directories_exist()
def get_engineered_data_path(date=None)
⋮----
date_str = date or datetime.now().strftime("%Y%m%d")
⋮----
CACHE_DIR = os.path.join(ENGINEERED_DATA_DIR, "cache")
⋮----
FEATURE_IMPORTANCE_FILE = os.path.join(ENGINEERED_DATA_DIR, "feature_importance.json")
def cleanup_cache(max_age_days=7, dry_run=False)
⋮----
now = time.time()
cache_files = glob.glob(os.path.join(CACHE_DIR, "*.pkl"))
deleted_count = 0
⋮----
# Get file modification time
mod_time = os.path.getmtime(cache_file)
# Calculate age in days
age_days = (now - mod_time) / (60 * 60 * 24)
# Delete if older than max_age_days
⋮----
# Dictionary of feature importance (populated from file if exists)
FEATURE_IMPORTANCE = {}
⋮----
FEATURE_IMPORTANCE = json.load(f)
⋮----
# Function timing decorator for performance monitoring
def time_function(func)
⋮----
@wraps(func)
    def wrapper(*args, **kwargs)
⋮----
start_time = time.time()
result = func(*args, **kwargs)
end_time = time.time()
elapsed_time = end_time - start_time
⋮----
# Cache decorator for expensive functions
def cache_result(cache_key, expire_days=1)
⋮----
def decorator(func)
⋮----
@wraps(func)
        def wrapper(*args, **kwargs)
⋮----
# Check if caching is explicitly disabled with _use_cache=False
use_cache = kwargs.pop('_use_cache', True)
# Create cache filename
cache_file = os.path.join(CACHE_DIR, f"{cache_key}.pkl")
# Check if cache exists and is fresh and we're allowed to use it
⋮----
age_days = (time.time() - mod_time) / (60 * 60 * 24)
⋮----
def load_feature_importance()
def save_feature_importance(importance_dict)
def get_important_features(threshold=0.01)
⋮----
importance = load_feature_importance()
⋮----
important_features = [f for f, score in importance.items() if score >= threshold]
⋮----
def update_feature_importance_from_model(model_file, feature_names=None, threshold=0.01)
⋮----
model = joblib.load(model_file)
importances = {}
⋮----
feature_imps = model.feature_importances_
feat_names = feature_names or getattr(model, 'feature_names_in_', None)
⋮----
num_estimators = len(model.estimators_)
all_importances = {}
⋮----
est_importances = est.feature_importances_
feat_names = feature_names or getattr(est, 'feature_names_in_', None)
⋮----
importances = {k: v for k, v in importances.items() if v >= threshold}
existing = load_feature_importance()
merged = {**existing, **importances}
⋮----
@time_function
@cache_result('matchup_features')
def create_matchup_features(game_data)
⋮----
df = game_data.copy()
⋮----
opponents = pd.Series(index=df.index, dtype='object')
home_mask = df['is_home']
home_indices = df[home_mask].index
⋮----
home_opponents = df.loc[home_mask, 'game_parts'].str[1]
⋮----
away_mask = ~df['is_home']
away_indices = df[away_mask].index
⋮----
away_opponents = df.loc[away_mask, 'game_parts'].str[0].str.split('_').str[1]
⋮----
df = df.drop(columns=['gameID_str', 'teamAbv_str', 'team_pattern', 'game_parts'])
⋮----
@time_function
@cache_result('defensive_matchup_features')
def create_defensive_matchup_features(game_data, team_ratings_path=None)
⋮----
team_def_ratings = {}
⋮----
team_ratings = pd.read_csv(team_ratings_path)
⋮----
median_rating = pd.Series(team_def_ratings.values()).median()
⋮----
min_rating = df['opp_defensive_rating'].min()
max_rating = df['opp_defensive_rating'].max()
⋮----
opp_def = df.groupby('opponent')['pts'].mean().reset_index()
opp_def = opp_def.rename(columns={'pts': 'avg_pts_allowed'})
df = pd.merge(df, opp_def, on='opponent', how='left')
max_pts = df['avg_pts_allowed'].max()
min_pts = df['avg_pts_allowed'].min()
⋮----
df = df.drop(columns=['avg_pts_allowed'])
⋮----
pos_groups = df.groupby(['opponent', 'position'])
⋮----
pos_stat = pos_groups[stat].mean().reset_index()
pos_stat = pos_stat.rename(columns={stat: f'opp_vs_{stat}_by_pos'})
league_pos_avg = df.groupby('position')[stat].mean().reset_index()
league_pos_avg = league_pos_avg.rename(columns={stat: f'league_avg_{stat}_by_pos'})
df = pd.merge(df, pos_stat, on=['opponent', 'position'], how='left')
df = pd.merge(df, league_pos_avg, on=['position'], how='left')
⋮----
df = df.drop(columns=[f'opp_vs_{stat}_by_pos', f'league_avg_{stat}_by_pos'])
shot_loc_cols = [col for col in df.columns if 'zone' in col.lower() and 'pct' in col.lower()]
⋮----
temp_df = df[['opponent', 'game_date', 'pts']].copy()
⋮----
team_games = temp_df[temp_df['opponent'] == team].sort_values('game_date')
⋮----
recent_def = team_games[['game_date', 'recent_def_pts']].dropna()
⋮----
prior_games = recent_def[recent_def['game_date'] < date]
⋮----
most_recent = prior_games.iloc[-1]['recent_def_pts']
⋮----
min_val = df['opp_recent_def'].min()
max_val = df['opp_recent_def'].max()
⋮----
df = df.drop(columns=['opp_recent_def'])
def_cols = [col for col in df.columns if 'def_' in col or 'opp_' in col]
⋮----
@time_function
@cache_result('rest_features')
def create_rest_features(game_data)
⋮----
df = df.sort_values(['playerID', 'game_date'])
⋮----
median_rest = df['days_rest'].median()
⋮----
conditions = [
⋮----
df['days_rest'] < 2,                           # Back-to-back
(df['days_rest'] >= 2) & (df['days_rest'] <= 3),  # Optimal rest
(df['days_rest'] > 3) & (df['days_rest'] <= 6),   # Longer rest
df['days_rest'] > 6                            # Very long rest
⋮----
choices = [-0.2, 0.1, 0.0, -0.1]
⋮----
# Back-to-back impact specific features (all vectorized)
⋮----
# Calculate games in last 7 days (simplified approach for robustness)
⋮----
df['games_last_7_days'] = 0  # Initialize with zeros
# Process each player group separately
⋮----
# Make sure dates are sorted
player_df = player_df.sort_values('game_date')
# Remove rows with NaT dates
valid_dates = player_df['game_date'].notna()
⋮----
player_df = player_df[valid_dates].copy()
⋮----
# For each game, count games in previous 7 days
⋮----
if i == 0:  # First game has 0 previous games
⋮----
# Get the date to look back from
current_date = row['game_date']
⋮----
# Define the cutoff date (7 days before)
cutoff_date = current_date - pd.Timedelta(days=7)
# Count games in the 7-day window
prev_games = player_df[
count = len(prev_games)
# Update the count in the main dataframe
⋮----
# Fill NaNs with 0 for first games
⋮----
# Create fatigue score using vectorized numpy select
fatigue_conditions = [
⋮----
df['games_last_7_days'] <= 2,  # Low fatigue
df['games_last_7_days'] == 3,  # Moderate fatigue
df['games_last_7_days'] == 4,  # High fatigue
df['games_last_7_days'] >= 5   # Very high fatigue
⋮----
fatigue_choices = [0, 0.3, 0.6, 1.0]
⋮----
# Drop temporary column
df = df.drop(columns=['is_game'])
⋮----
@time_function
@cache_result('trend_features')
def create_trend_features(game_data, selective=True, use_weighted_averages=True)
⋮----
# Make a copy of the input DataFrame
⋮----
# Sort by player and date
⋮----
# Extract date from gameID (format: YYYYMMDD_TEAM@TEAM)
⋮----
# Get the list of stats to compute trends for
all_stat_cols = ['pts', 'reb', 'ast', 'stl', 'blk', 'TOV_x', 'fgm', 'fga', 'fgp',
# Only include columns that actually exist in the dataframe
available_cols = [col for col in all_stat_cols if col in df.columns]
# If selective is True, use feature importance to pick most significant stats
⋮----
# Get feature importance from saved data
⋮----
# Determine which statistics have high importance in trend features
# Look for any trend features in the importance data
trend_pattern = re.compile(r'(last3_|last10_|_trend|_w_avg)')
trend_importances = {k: v for k, v in importance.items() if trend_pattern.search(k)}
⋮----
# Extract the base stat names from important trend features
base_stats = set()
⋮----
# Remove prefixes/suffixes to get the base stat name
base_stat = re.sub(r'last3_|last10_|_w_avg|_avg|_trend', '', feature)
⋮----
# Filter to only important base stats that are available
stat_cols = [col for col in available_cols if col in base_stats]
# If we don't have any important stats, fall back to primary stats
⋮----
stat_cols = ['pts', 'reb', 'ast', 'mins']
⋮----
stat_cols = [col for col in stat_cols if col in available_cols]
⋮----
stat_cols = available_cols
⋮----
# Create all trend features efficiently in a single pass
# Group by playerID first to avoid multiple groupby operations
player_groups = df.groupby('playerID')
# Calculate rolling averages for last 3 and last 10 games for each stat
⋮----
# Standard unweighted averages
# Last 3 games average (vectorized)
⋮----
# Last 10 games average (vectorized)
⋮----
# Weighted averages (if enabled)
⋮----
# Define a function for exponentially weighted average calculation
# This gives higher weight to more recent games
# Exponentially weighted moving average for last 3 games (higher weight to recent games)
⋮----
# Exponentially weighted moving average for last 10 games
⋮----
# Calculate trends using weighted averages
⋮----
# Calculate trend (difference between short and medium term)
⋮----
# Fill all NaN values with 0 (vectorized)
trend_cols = [col for col in df.columns if (
⋮----
@time_function
@cache_result('time_weighted_features')
def create_time_weighted_features(game_data, decay_factor=0.9, max_games=20)
⋮----
# Try to extract date from gameID
⋮----
# Get the list of stats to compute time-weighted features for
stat_cols = ['pts', 'reb', 'ast', 'stl', 'blk', 'fgm', 'fga', 'tptfgm', 'tptfga', 'mins']
⋮----
available_stats = [col for col in stat_cols if col in df.columns]
# Initialize feature tracking
time_weighted_cols = []
# Create a DataFrame to store results to avoid repeated column creation
result_df = pd.DataFrame(index=df.index)
# Process each player separately to apply time-weighted calculations
⋮----
# Skip if player has too few games
⋮----
# Sort by date
⋮----
player_dates = player_df['game_date'].values
# Process each game for this player
⋮----
# Get all games before current game
past_games = player_df[player_df['game_date'] < current_date]
# Skip if not enough past games
⋮----
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
⋮----
# Get past values
past_values = past_games[stat].values
# Calculate weighted average
weighted_avg = np.sum(past_values * weights)
# Create column name
col_name = f'tw_{stat}'
# Add to results
⋮----
# Track columns created
⋮----
# Create exponential moving average features (EMA)
# These give higher weight to recent games but with smoother transitions
⋮----
# Use pandas built-in EMA calculation
# Try different alpha values (smoothing factors)
for alpha in [0.3, 0.7]:  # 0.3 = more smoothing, 0.7 = more responsive
# Create temp series with past values plus current value
temp_series = pd.Series(
# Calculate EMA - drop the last value (current game)
ema_values = temp_series.ewm(alpha=alpha).mean().iloc[:-1].values
# Only use if we have enough values
⋮----
# Column name encodes the alpha value
alpha_str = str(int(alpha * 10))
col_name = f'ema{alpha_str}_{stat}'
# Store the last EMA value (most recent)
⋮----
# Track columns created
⋮----
# Create momentum indicators (direction of recent performance)
⋮----
# Need at least 5 games to calculate meaningful momentum
⋮----
# Recent 3 games average
recent_3 = past_games.sort_values('game_date', ascending=False).head(3)[stat].mean()
# Previous 5 games average (excluding most recent 3)
prev_games = past_games.sort_values('game_date', ascending=False)
⋮----
prev_5 = prev_games.iloc[3:8][stat].mean() if len(prev_games) >= 8 else prev_games.iloc[3:][stat].mean()
# Calculate momentum (ratio of recent to previous performance)
momentum = recent_3 / prev_5 if prev_5 > 0 else 1.0
col_name = f'momentum_{stat}'
⋮----
# Add interpretation column - categorical momentum indicator
col_name_cat = f'momentum_{stat}_cat'
⋮----
result_df.loc[idx, col_name_cat] = 2  # Strong positive momentum
⋮----
result_df.loc[idx, col_name_cat] = 1  # Positive momentum
⋮----
result_df.loc[idx, col_name_cat] = -2  # Strong negative momentum
⋮----
result_df.loc[idx, col_name_cat] = -1  # Negative momentum
⋮----
result_df.loc[idx, col_name_cat] = 0  # Stable performance
# Track columns
⋮----
# Merge results back to main dataframe
df = pd.concat([df, result_df], axis=1)
# Fill NaN values
⋮----
# For categorical columns, fill with 0 (neutral)
⋮----
# For continuous columns, use median of that column
⋮----
def create_opp_strength_features(game_data, team_ratings_path=None)
⋮----
# Try to find the latest team ratings file
latest_date = datetime.now().strftime("%Y%m%d")
team_ratings_path = f"/Users/lukesmac/Projects/nbaModel/data/standings/team_ratings_{latest_date}.csv"
# If the file doesn't exist, use a placeholder approach
⋮----
strong_teams = ['BOS', 'DEN', 'LAL', 'MIL', 'PHI', 'GSW', 'MIA']
medium_teams = ['DAL', 'NYK', 'LAC', 'PHO', 'CLE', 'MEM', 'SAC', 'NOP', 'MIN']
weak_teams = ['OKC', 'ATL', 'BRK', 'TOR', 'CHI', 'WAS', 'UTA', 'POR', 'SAS', 'CHO', 'ORL', 'IND', 'HOU', 'DET']
⋮----
team_mapping = {
⋮----
rating_cols = ['Offensive_Rating', 'Defensive_Rating', 'Net_Rating', 'W', 'L']
rating_cols = [col for col in rating_cols if col in team_ratings.columns]
team_rating_dict = {}
⋮----
team_abbr = row['Team_Abbr']
⋮----
min_rating = df['opp_Net_Rating'].min()
max_rating = df['opp_Net_Rating'].max()
⋮----
@time_function
@cache_result('consistency_features')
def create_player_consistency_features(game_data, selective=True)
⋮----
all_stat_cols = ['pts', 'reb', 'ast', 'mins']
⋮----
consistency_pattern = re.compile(r'(_consistency)')
consistency_importances = {k: v for k, v in importance.items() if consistency_pattern.search(k)}
⋮----
base_stat = re.sub(r'_consistency', '', feature)
⋮----
# Just use points consistency at minimum as it's typically most important
stat_cols = ['pts'] if 'pts' in available_cols else available_cols[:1]
⋮----
# Group by playerID to avoid multiple groupby operations
⋮----
# Calculate consistency metrics for each selected stat
⋮----
# Calculate rolling mean and std for last 10 games (vectorized)
roll_mean = player_groups[stat].rolling(window=10, min_periods=5).mean().reset_index(level=0, drop=True).shift(1)
roll_std = player_groups[stat].rolling(window=10, min_periods=5).std().reset_index(level=0, drop=True).shift(1)
# Calculate CV (with handling for division by zero)
⋮----
# Fill NaNs, then invert the CV so higher values mean more consistency
⋮----
# Create an overall consistency score (average of individual consistency scores)
consistency_cols = [f'{stat}_consistency' for stat in stat_cols]
if consistency_cols:  # Only if we have any consistency columns
⋮----
def create_advanced_offensive_features(game_data)
⋮----
# Calculate eFG% (Effective Field Goal %) if not already present
# Formula: eFG% = (FGM + 0.5 * 3PTM) / FGA
⋮----
# Check if we have the necessary columns
⋮----
# Handle division by zero
mask = df['fga'] > 0
⋮----
# Calculate TS% (True Shooting %) if not already present
# Formula: TS% = PTS / (2 * (FGA + 0.44 * FTA))
⋮----
denominator = 2 * (df['fga'] + 0.44 * df['fta'])
mask = denominator > 0
⋮----
# Calculate Assist-to-Turnover ratio
⋮----
mask = df['TOV_x'] > 0
⋮----
# For games with 0 turnovers but positive assists, set a high ratio
mask_zero_tov = (df['TOV_x'] == 0) & (df['ast'] > 0)
df.loc[mask_zero_tov, 'AST_TO_ratio'] = df.loc[mask_zero_tov, 'ast'] * 2  # A reasonable high value
# For games with 0 assists and 0 turnovers, set to 1 (neutral)
mask_both_zero = (df['TOV_x'] == 0) & (df['ast'] == 0)
⋮----
# Points per Shot (PPS) - another useful efficiency metric
⋮----
# Create relative offensive efficiency metrics compared to team or league average
# (if team averages are available)
⋮----
def create_usage_features(game_data)
⋮----
# Check if we have USG% from season averages
⋮----
# No transformation needed, already exists
⋮----
# Rename to standard format
⋮----
# Try to calculate a simplified usage approximation from game data
⋮----
# Simple formula: (FGA + 0.44*FTA + TOV) / minutes
⋮----
# Scale to a percentage format (0-100)
max_usg = df['approx_usg'].max()
⋮----
# Create usage-related features
⋮----
# Usage categories
⋮----
# Calculate recent usage trend (last 5 games vs season average)
⋮----
# We can estimate recent usage with available stats
⋮----
# Create usage-efficiency interaction
⋮----
def create_player_specific_factors(game_data)
⋮----
# Detect potential minutes restrictions by comparing to season averages
# This is an approximation as we don't have actual injury data
⋮----
player_median_mins = df.groupby('playerID')['mins'].median().reset_index()
⋮----
# Merge back to the main dataframe
df = pd.merge(df, player_median_mins, on='playerID', how='left')
# Calculate minutes deviation from typical minutes
⋮----
# Flag possible minutes restriction
# Consider a player on minutes restriction if they played significantly less than their median
# but still played in the game (e.g., more than 5 minutes)
⋮----
# Also flag games where minutes are returning to normal after a restriction
# This helps identify players coming back from injury
⋮----
# Create a feature for recent minutes volatility
# High volatility might indicate a player with changing role or returning from injury
⋮----
# Detect mid-season team changes
⋮----
# For each player, check if their team changes from one game to the next
⋮----
# Count games since team change (useful for adaptation period)
# Reset counter at each team change
⋮----
# Process each player separately
⋮----
continue  # No team changes for this player
# Initialize counter
counter = 0
indices = player_df.index
⋮----
if i == 0:  # First game, no previous team
⋮----
# Reset counter at team change
counter = 1
⋮----
# Increment counter
⋮----
# Create a feature that indicates the adaptation period after a trade
# (first 10 games with new team are typically an adjustment period)
⋮----
# Create features for back-to-back games which can indicate fatigue
# This is already handled in the rest features, but we'll add it here for completeness
⋮----
young_mask = (df['Age'] < 25) & (df['days_rest'] < 2)
⋮----
mid_mask = (df['Age'] >= 25) & (df['Age'] <= 32) & (df['days_rest'] < 2)
⋮----
old_mask = (df['Age'] > 32) & (df['days_rest'] < 2)
⋮----
@time_function
@cache_result('team_lineup_features')
def create_team_lineup_features(game_data)
⋮----
has_starter_info = 'starter' in df.columns or 'gameStarter' in df.columns
starter_col = 'starter' if 'starter' in df.columns else 'gameStarter' if 'gameStarter' in df.columns else None
⋮----
team_df = team_df.sort_values('game_date')
⋮----
past_games = team_df[team_df['game_date'] < date]
⋮----
recent_games = past_games.sort_values('game_date', ascending=False).head(10)
⋮----
game_starters = recent_games[recent_games['is_starter'] == 1].groupby('game_date')['playerID'].apply(list)
lineup_counts = {}
⋮----
key = tuple(sorted(starter_list))
⋮----
max_count = max(lineup_counts.values())
continuity = max_count / len(game_starters)
current_indices = date_df.index
⋮----
player_id = player_row['playerID']
player_past = recent_games[recent_games['playerID'] == player_id]
⋮----
mins_mean = player_past['mins'].mean()
mins_std = player_past['mins'].std()
⋮----
cv = mins_std / mins_mean
consistency = 1 / (1 + cv) if cv > 0 else 1.0
⋮----
player_game_dates = player_past['game_date'].unique()
⋮----
# Get all teammates in these games
teammates = recent_games[
# Find current teammates in this game
current_teammates = date_df[date_df['playerID'] != player_id]['playerID'].unique()
# Calculate how many current teammates are familiar to this player
⋮----
familiar_teammates = teammates['playerID'].unique()
overlap_count = sum(1 for t in current_teammates if t in familiar_teammates)
chemistry = overlap_count / len(current_teammates)
⋮----
# Calculate team cohesion based on player tenure with team
⋮----
# For each player, calculate how many games they've played with their current team
player_team_counts = {}
⋮----
key = (team, player)
current_count = player_team_counts.get(key, 0)
⋮----
max_games = df['player_team_games'].max()
⋮----
df = df.drop(columns=['player_team_games'])
lineup_cols = ['team_lineup_continuity', 'player_role_consistency',
⋮----
def create_feature_matrix(processed_data, features_to_use=None, target_cols=None)
⋮----
df = processed_data.copy()
⋮----
target_candidates = [
best_match = []
⋮----
matches = [col for col in candidates if col in df.columns]
⋮----
best_match = matches
target_cols = best_match
⋮----
stat_pattern = re.compile(r'(pts|points|reb|rebounds|ast|assists|stl|steals|blk|blocks|tov|turnovers)', re.IGNORECASE)
target_cols = [col for col in df.columns if stat_pattern.search(col)]
⋮----
profile_cols = ['MP', 'Age', 'home_game', 'days_rest', 'USG%', 'WS/48', 'OBPM', 'DBPM', 'BPM', 'opp_strength']
season_avg_cols = ['PTS', 'TRB', 'AST', 'STL', 'BLK', 'TOV_y', 'FG%', '3P%', 'FT%']
advanced_cols = ['eFG%', 'TS%', 'AST_TO_ratio', 'PPS']
recent_cols = [col for col in df.columns if (
⋮----
consistency_cols = [col for col in df.columns if col.endswith('_consistency')]
player_specific_cols = [
std_cols = [col for col in df.columns if col.startswith('std_')]
all_feature_cols = (
features_to_use = [col for col in all_feature_cols if col in df.columns]
⋮----
features_to_use = [col for col in df.columns if col not in target_cols]
df = df.dropna(subset=target_cols)
X = df[features_to_use].copy()
y = df[target_cols].copy()
⋮----
numeric_cols = X.select_dtypes(include=['int', 'float']).columns
⋮----
na_mask = X[col].isna()
⋮----
non_numeric_cols = X.select_dtypes(exclude=['int', 'float']).columns
⋮----
mode_values = X[col].mode()
⋮----
valid_rows = ~X.isnull().any(axis=1)
X = X.loc[valid_rows]
y = y.loc[valid_rows]
feature_names = X.columns.tolist()
⋮----
empty_X = pd.DataFrame()
empty_y = pd.DataFrame(columns=target_cols if target_cols else [])
⋮----
@time_function
def detect_and_handle_redundant_features(df, importance_threshold=0.01, correlation_threshold=0.9, verbose=True)
⋮----
df_copy = df.copy()
⋮----
redundant_patterns = [
to_drop = []
drop_reasons = {}
⋮----
# Get importance scores (default to 0 if not found)
imp1 = importance.get(col1, 0)
imp2 = importance.get(col2, 0)
# If both are below threshold, drop the second one
⋮----
# Calculate correlation if both columns have numeric data
⋮----
correlation = df_copy[col1].corr(df_copy[col2])
# If correlation is NaN, skip this pair
⋮----
# If highly correlated, keep the one with higher importance
⋮----
# Find highly correlated features more generally
numeric_df = df_copy.select_dtypes(include=['float64', 'float32', 'int64', 'int32'])
# Skip the correlation calculation if the dataframe is too big
if numeric_df.shape[1] < 100:  # Only run on dataframes with reasonable number of columns
⋮----
# Calculate correlation matrix
corr_matrix = numeric_df.corr()
# Get upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
# Find features with correlation greater than threshold
high_corr_pairs = []
⋮----
# Get highly correlated features for this column
correlated_features = upper.index[upper[col].abs() > correlation_threshold].tolist()
⋮----
if col != corr_feat:  # Avoid self-correlation (though not needed with upper triangle)
⋮----
# For each pair, decide which to drop based on feature importance
⋮----
# Skip if either is already in to_drop
⋮----
# Drop the one with lower importance
⋮----
# Remove duplicates from to_drop
to_drop = list(set(to_drop))
# Log the dropped columns and reasons
⋮----
# Drop the columns
df_copy = df_copy.drop(columns=to_drop, errors='ignore')
⋮----
def detect_unused_features(df, importance_threshold=0.005, min_pct_drop=0.1, verbose=True)
⋮----
# Make a copy of the input dataframe
⋮----
# Get current feature importance data
⋮----
# Always keep certain columns regardless of importance
keep_columns = [
⋮----
'pts', 'reb', 'ast', 'stl', 'blk', 'TOV_x', 'plusMinus'  # Target columns
⋮----
# Identify features with importance below threshold
low_importance = {}
⋮----
# Skip columns we always want to keep
⋮----
# Get importance score (default to 0 if not found)
imp = importance.get(col, 0)
# If below threshold, mark for potential removal
⋮----
# Sort by importance (ascending)
low_importance = {k: v for k, v in sorted(low_importance.items(), key=lambda item: item[1])}
# Determine how many columns to drop (at least min_pct_drop of total)
min_drop_count = max(int(len(df_copy.columns) * min_pct_drop), 1)
drop_count = min(len(low_importance), min_drop_count)
# Select columns to drop (lowest importance first)
to_drop = list(low_importance.keys())[:drop_count]
# Log the dropped columns
⋮----
def validate_derived_features(df, verbose=True)
⋮----
# Define validation checks for different statistical categories
validations = {
⋮----
# Per-36 stats should be within reasonable ranges
⋮----
('pts_per36', 0, 60),  # Max ~60 pts per 36 min is reasonable
('reb_per36', 0, 40),  # Max ~40 reb per 36 min is reasonable
('ast_per36', 0, 30),  # Max ~30 ast per 36 min is reasonable
('stl_per36', 0, 15),  # Max ~15 stl per 36 min is reasonable
('blk_per36', 0, 15),  # Max ~15 blk per 36 min is reasonable
('TOV_x_per36', 0, 20)  # Max ~20 tov per 36 min is reasonable
⋮----
# Efficiency metrics
⋮----
('eFG%', 0, 1),  # Effective field goal % should be 0-1
('TS%', 0, 1),   # True shooting % should be 0-1
('PPS', 0, 3),   # Points per shot typically 0-3
('AST_TO_ratio', 0, 20)  # Assist to turnover ratio rarely above 20
⋮----
# Trend features
⋮----
'checks': []  # Will be dynamically populated
⋮----
# Weighted average features
⋮----
# Consistency features
⋮----
('overall_consistency', 0, 1),  # Consistency score should be 0-1
('pts_consistency', 0, 1),      # Consistency score should be 0-1
('reb_consistency', 0, 1),      # Consistency score should be 0-1
('ast_consistency', 0, 1)       # Consistency score should be 0-1
⋮----
# Find all columns matching patterns
⋮----
if not config['cols']:  # If not explicitly specified
pattern = re.compile(config['pattern'])
⋮----
# Add dynamic checks for trend features
trend_cols = validations['trend']['cols']
⋮----
# Base stat from trend column name
base_stat = col.replace('_trend', '')
base_stat = base_stat.replace('_w', '')
# Skip non-numeric columns
⋮----
# Calculate reasonable min/max based on the standard deviation of the base stat
std_val = df_copy[base_stat].std()
⋮----
min_val = -5 * std_val  # 5 standard deviations below
max_val = 5 * std_val   # 5 standard deviations above
⋮----
# Add dynamic checks for weighted average features
w_avg_cols = validations['w_avg']['cols']
⋮----
# Base stat from weighted average column name
base_stat = col.replace('_w_avg', '')
⋮----
base_stat = base_stat.replace('last3_', '')
⋮----
base_stat = base_stat.replace('last10_', '')
⋮----
# Calculate reasonable min/max based on the min/max of the base stat
min_val = df_copy[base_stat].min() if pd.notna(df_copy[base_stat].min()) else 0
max_val = df_copy[base_stat].max() if pd.notna(df_copy[base_stat].max()) else 100
⋮----
# Track validation issues
validation_issues = {}
# Apply all validation checks
⋮----
# Count out-of-range values
below_min = (df_copy[col] < min_val).sum()
above_max = (df_copy[col] > max_val).sum()
# Log and fix issues
⋮----
# Clip values to valid range
⋮----
# Log validation results
⋮----
def check_cache_freshness(cache_file, input_data=None, recache_threshold_days=3)
⋮----
# If file doesn't exist, it needs to be generated
⋮----
cached_data = pickle.load(f)
⋮----
use_cache = False
save_to_cache = False
⋮----
date_str = datetime.now().strftime("%Y%m%d")
row_count = len(processed_data)
cache_key = f"all_features_{date_str}_{row_count}"
⋮----
df = pickle.load(f)
⋮----
def create_matchup_unwrapped(data)
def create_rest_unwrapped(data)
def create_trend_unwrapped(data, selective=True, use_weighted_avgs=True)
def create_consistency_unwrapped(data, selective=True)
def create_defensive_matchup_unwrapped(data, team_ratings_path=None)
def create_time_weighted_unwrapped(data, decay_factor=0.9, max_games=20)
def create_team_lineup_unwrapped(data)
⋮----
processed_data = fix_invalid_values(processed_data)
processed_data = resolve_turnover_columns(processed_data)
df = create_matchup_unwrapped(processed_data)
df = create_rest_unwrapped(df)
df = create_advanced_offensive_features(df)
df = create_opp_strength_features(df, team_ratings_path)
df = create_player_specific_factors(df)
df = create_trend_unwrapped(df, selective=selective, use_weighted_avgs=use_weighted_averages)
df = create_consistency_unwrapped(df, selective=selective)
team_ratings_path = os.path.join(DATA_DIR, "standings", f"team_ratings_{datetime.now().strftime('%Y%m%d')}.csv")
⋮----
yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y%m%d')
team_ratings_path = os.path.join(DATA_DIR, "standings", f"team_ratings_{yesterday}.csv")
df = create_defensive_matchup_unwrapped(df, team_ratings_path=team_ratings_path)
df = create_time_weighted_unwrapped(df)
df = create_team_lineup_unwrapped(df)
df = create_usage_features(df)
⋮----
df = validate_derived_features(df)
⋮----
df = detect_and_handle_redundant_features(df)
df = detect_unused_features(df)
⋮----
keep_cols = [
pattern_cols = [
all_keep_cols = keep_cols + pattern_cols
# but exclude playerID, teamID, and other unnecessary IDs
⋮----
# Keep target columns for training
target_cols = ['pts', 'reb', 'ast', 'stl', 'blk', 'TOV_x', 'plusMinus']
⋮----
# Get unique column list
all_keep_cols = list(set(all_keep_cols))
# Check which columns exist in our dataframe
existing_cols = [col for col in all_keep_cols if col in df.columns]
# Create a copy with only the columns we want to keep
columns_to_drop = [col for col in df.columns if col not in existing_cols]
⋮----
df = df[existing_cols].copy()
# Save to cache if requested
⋮----
# Define command line arguments for flexible feature engineering
parser = argparse.ArgumentParser(description="Generate engineered features for NBA player performance prediction")
# Data input/output options
input_group = parser.add_argument_group('Data Options')
⋮----
# Feature generation options
feature_group = parser.add_argument_group('Feature Options')
⋮----
# Data quality options
quality_group = parser.add_argument_group('Data Quality Options')
⋮----
# Caching options
cache_group = parser.add_argument_group('Caching Options')
⋮----
# Performance options
perf_group = parser.add_argument_group('Performance Options')
⋮----
args = parser.parse_args()
# Ensure directories exist
⋮----
# Check if we should just clean up the cache
⋮----
# Just run cache cleanup
deleted_count = cleanup_cache(max_age_days=args.cache_max_age, dry_run=args.cache_dry_run)
⋮----
# Check if we should just update feature importance
⋮----
# Just update feature importance from model
updated = update_feature_importance_from_model(
⋮----
num_features = len(updated)
⋮----
top_features = sorted(updated.items(), key=lambda x: x[1], reverse=True)[:10]
⋮----
# Get the latest processed data file if not specified
⋮----
processed_data_path = args.input_file
⋮----
# Try to load from config, otherwise use default path
⋮----
processed_data_path = get_processed_data_path()
⋮----
# Default fallback
current_date = datetime.now().strftime("%Y%m%d")
processed_data_path = f"/Users/lukesmac/Projects/nbaModel/data/processed/processed_nba_data_{current_date}.csv"
# If today's file doesn't exist, try to find the latest one
⋮----
processed_dir = os.path.dirname(processed_data_path)
⋮----
processed_files = sorted(
⋮----
processed_data_path = os.path.join(processed_dir, processed_files[0])
⋮----
# Clean up cache if requested (before loading data)
⋮----
# Load the processed data
⋮----
processed_data = pd.read_csv(processed_data_path)
⋮----
# Apply data quality checks if requested and available
⋮----
processed_data = run_all_quality_checks(processed_data, min_minutes=args.min_minutes)
⋮----
# Apply feature engineering
engineered_data = engineer_all_features(
# Determine output path
⋮----
output_path = args.output_file
⋮----
output_path = get_engineered_data_path()
⋮----
output_dir = "/Users/lukesmac/Projects/nbaModel/data/engineered"
⋮----
output_path = os.path.join(output_dir, f"engineered_nba_data_{current_date}.csv")
# Save engineered data
⋮----
# Print summary of features generated
feature_count = engineered_data.shape[1] - processed_data.shape[1]
⋮----
# If profiling was requested, print detailed timing information
⋮----
# Sort functions by execution time (if available in globals)
timing_info = {}
⋮----
# Print timing info
````

## File: src/feature_viz.py
````python
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
ENGINEERED_DATA_DIR = os.path.join(DATA_DIR, "engineered")
FEATURE_IMPORTANCE_FILE = os.path.join(ENGINEERED_DATA_DIR, "feature_importance.json")
VIZ_OUTPUT_DIR = os.path.join(DATA_DIR, "visualizations")
⋮----
def load_feature_importance(file_path=None)
⋮----
file_path = FEATURE_IMPORTANCE_FILE
⋮----
data = json.load(f)
⋮----
flat_importance = {}
⋮----
feature = item["feature"]
importance = item["importance_mean"]
⋮----
importance = item["importance"]
⋮----
data = data["feature_importance"]
⋮----
def load_historical_importance(dir_path=None)
⋮----
models_dir = os.path.join(os.path.dirname(DATA_DIR), "models")
⋮----
models_dir = dir_path
# Find all feature importance JSON files
importance_files = []
⋮----
# Also try to find metrics files which may contain feature importance
⋮----
# Load all files and combine data
all_data = []
⋮----
# Parse date from filename
date_str = None
⋮----
date_str = os.path.basename(file_path).replace('feature_importance_', '').replace('.json', '')
⋮----
# Try to extract date from metrics filename
parts = os.path.basename(file_path).split('_')
⋮----
if part.isdigit() and len(part) == 8:  # YYYYMMDD format
date_str = part
⋮----
# Use file modification time if date not found in filename
mod_time = os.path.getmtime(file_path)
date_str = datetime.fromtimestamp(mod_time).strftime("%Y%m%d")
# Convert to datetime
⋮----
date = datetime.strptime(date_str, "%Y%m%d")
⋮----
# If parsing fails, use file modification time
⋮----
date = datetime.fromtimestamp(mod_time)
# Load feature importance data
⋮----
# Handle different JSON formats
# Check if data has the new structure with target_importances and permutation_importance
⋮----
# Extract permutation importance which is more reliable
⋮----
# Try to get permutation importance for all targets
⋮----
# Use highest importance value if feature appears in multiple targets
⋮----
# If no permutation importance data, try target_importances
⋮----
# Use highest importance value if feature appears in multiple targets
⋮----
# Convert to DataFrame format
⋮----
# Find feature importance in metrics file if needed
⋮----
# Standard flat dictionary format
⋮----
# Create DataFrame
df = pd.DataFrame(all_data)
⋮----
def plot_top_features(importance_data, top_n=20, output_file=None, show_plot=True)
⋮----
# Convert dictionary to sorted list of (feature, importance) tuples
sorted_features = sorted(importance_data.items(), key=lambda x: x[1], reverse=True)[:top_n]
# Create Series for plotting
feature_names = [f[0] for f in sorted_features]
importance_values = [f[1] for f in sorted_features]
# Plot horizontal bar chart
⋮----
# Get the latest date data
latest_date = importance_data['date'].max()
latest_data = importance_data[importance_data['date'] == latest_date]
# Get top N features
top_features = latest_data.nlargest(top_n, 'importance')
# Plot using seaborn for better appearance
⋮----
def plot_feature_importance_history(history_df, top_n=10, output_file=None, show_plot=True)
⋮----
# Get top N features across all time points
overall_top_features = history_df.groupby('feature')['importance'].mean().nlargest(top_n).index.tolist()
# Filter data to only include these features
plot_data = history_df[history_df['feature'].isin(overall_top_features)]
⋮----
# Plot time series for each feature
pivot_data = plot_data.pivot(index='date', columns='feature', values='importance')
⋮----
# Format x-axis as dates
⋮----
# Add legend outside the plot area
⋮----
def plot_feature_category_importance(importance_data, output_file=None, show_plot=True)
⋮----
# Define categories and their patterns
categories = {
# Convert data to dictionary if needed
⋮----
latest_data = importance_data
# Convert to dictionary
importance_dict = dict(zip(latest_data['feature'], latest_data['importance']))
⋮----
importance_dict = importance_data
# Calculate category importance
category_importance = {}
uncategorized_importance = 0
⋮----
categorized = False
⋮----
categorized = True
⋮----
# Add uncategorized if there are any
⋮----
# Create pie chart
⋮----
# Sort categories by importance
sorted_categories = sorted(category_importance.items(), key=lambda x: x[1], reverse=True)
labels = [cat[0] for cat in sorted_categories]
sizes = [cat[1] for cat in sorted_categories]
# Plot with percentage labels
⋮----
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
⋮----
def create_all_visualizations(importance_file=None, history_dir=None, output_dir=None, show_plots=False)
⋮----
# Set default output dir if not provided
⋮----
output_dir = VIZ_OUTPUT_DIR
⋮----
# Generate timestamp for filenames
timestamp = datetime.now().strftime("%Y%m%d")
# Load current feature importance
importance_data = load_feature_importance(importance_file)
⋮----
# Plot top features
top_features_file = os.path.join(output_dir, f"top_features_{timestamp}.png")
⋮----
# Plot feature categories
categories_file = os.path.join(output_dir, f"feature_categories_{timestamp}.png")
⋮----
# Load and plot historical data if available
history_df = load_historical_importance(history_dir)
⋮----
history_file = os.path.join(output_dir, f"feature_history_{timestamp}.png")
⋮----
parser = argparse.ArgumentParser(description='Feature Importance Visualization Tool')
# Input options
⋮----
# Output options
⋮----
# Visualization options
⋮----
# Specific plot options
⋮----
args = parser.parse_args()
⋮----
importance_data = load_feature_importance(args.importance_file)
⋮----
output_dir = args.output_dir or VIZ_OUTPUT_DIR
⋮----
history_df = load_historical_importance(args.history_dir)
````

## File: src/memory_utils.py
````python
TQDM_AVAILABLE = True
⋮----
TQDM_AVAILABLE = False
⋮----
def get_memory_usage()
⋮----
process = psutil.Process(os.getpid())
mem_info = process.memory_info()
⋮----
def memory_usage_report()
⋮----
mem_usage = get_memory_usage()
⋮----
def profile_memory(func)
⋮----
@wraps(func)
    def wrapper(*args, **kwargs)
⋮----
mem_before = get_memory_usage()
⋮----
result = func(*args, **kwargs)
⋮----
mem_after = get_memory_usage()
⋮----
def optimize_dataframe(df, categorical_threshold=10, datetime_cols=None, verbose=False)
⋮----
result = df.copy()
initial_memory = result.memory_usage(deep=True).sum() / (1024 * 1024)
⋮----
col_type = result[col].dtype
⋮----
col_min = result[col].min()
col_max = result[col].max()
⋮----
num_unique = result[col].nunique()
total_values = len(result[col])
⋮----
final_memory = result.memory_usage(deep=True).sum() / (1024 * 1024)
savings = 100 * (1 - final_memory / initial_memory)
⋮----
class ProgressLogger
⋮----
def update(self, n: int = 1)
⋮----
current_time = time.time()
⋮----
percentage = min(100.0, 100.0 * self.n / self.total if self.total > 0 else 100.0)
⋮----
def set_description(self, desc: str)
def close(self)
def __enter__(self)
def __exit__(self, exc_type, exc_val, exc_tb)
⋮----
results = []
⋮----
result = func(item, **kwargs)
⋮----
n_batches = (len(data_list) + batch_size - 1) // batch_size
⋮----
batch_start = i * batch_size
batch_end = min(batch_start + batch_size, len(data_list))
batch = data_list[batch_start:batch_end]
⋮----
batch_result = func(batch, *args, **kwargs)
⋮----
class PipelineTracker
⋮----
def __init__(self, stages: List[str], use_tqdm: bool = True)
def start_stage(self, stage_name: str, total_items: int = 0, unit: str = "it")
⋮----
stage_index = self.stages.index(stage_name)
⋮----
def update_stage(self, n: int = 1)
def complete_stage(self)
⋮----
stage_name = self.stages[self.current_stage] if self.current_stage < len(self.stages) else "Unknown stage"
⋮----
def finish(self, success: bool = True)
⋮----
mem = memory_usage_report()
⋮----
pipeline = PipelineTracker(stages=["Data Processing", "Feature Engineering", "Model Training", "Evaluation"])
stage1_progress = pipeline.start_stage("Data Processing", total_items=5)
⋮----
stage2_progress = pipeline.start_stage("Feature Engineering", total_items=3)
⋮----
def square(x)
results = progress_map(square, list(range(10)), desc="Squaring numbers", unit="number")
````

## File: src/model_builder.py
````python
BAYESIAN_OPT_AVAILABLE = True
⋮----
BAYESIAN_OPT_AVAILABLE = False
⋮----
XGBOOST_AVAILABLE = True
⋮----
XGBOOST_AVAILABLE = False
⋮----
LIGHTGBM_AVAILABLE = True
⋮----
LIGHTGBM_AVAILABLE = False
⋮----
def load_training_data(data_path=None)
⋮----
current_date = datetime.now().strftime("%Y%m%d")
engineered_dir = "/Users/lukesmac/Projects/nbaModel/data/engineered"
⋮----
# Path for today's engineered data
engineered_path = os.path.join(engineered_dir, f"engineered_nba_data_{current_date}.csv")
⋮----
engineered_files = []
⋮----
engineered_files = sorted(
⋮----
engineered_path = os.path.join(engineered_dir, engineered_files[0])
⋮----
processed_dir = "/Users/lukesmac/Projects/nbaModel/data/processed"
⋮----
processed_files = sorted(
⋮----
processed_path = os.path.join(processed_dir, processed_files[0])
⋮----
processed_data = pd.read_csv(processed_path)
engineered_data = engineer_all_features(processed_data)
⋮----
data_path = engineered_path
⋮----
data = pd.read_csv(data_path)
⋮----
hyperparams = {
⋮----
hyperparams = {}
⋮----
is_multi_output = isinstance(y_train, (pd.DataFrame, np.ndarray)) and y_train.shape[1] > 1
# Check if we should use ensemble stacking for the chosen model type
⋮----
# Use specialized stacked ensemble with emphasis on the chosen model type
model = create_specialized_ensemble(X_train, y_train, specialized_model=model_type,
# If multi-output, wrap in MultiOutputRegressor
⋮----
model = MultiOutputRegressor(model)
# Train the specialized stacked model
⋮----
# No feature importances for stacked models
⋮----
# Create the model based on model type
⋮----
base_model = DecisionTreeRegressor(**hyperparams)
⋮----
base_model = RandomForestRegressor(**hyperparams)
⋮----
base_model = GradientBoostingRegressor(**hyperparams)
⋮----
base_model = xgb.XGBRegressor(**hyperparams)
⋮----
base_model = lgb.LGBMRegressor(**hyperparams)
⋮----
# Create a standard stacked ensemble
model = create_stacked_ensemble(X_train, y_train, low_resource_mode=low_resource_mode,
⋮----
# Train the stacked model
⋮----
# Only usable for multi-output targets
⋮----
# Create target-specific model
model = TargetSpecificModel(target_models=target_specific_models, fallback_model='random_forest')
# Train the model
⋮----
# Get feature importances
feature_importances = model.get_feature_importance()
# Average feature importances across targets
⋮----
# Filter out None values
valid_importances = [imp for imp in feature_importances.values() if imp is not None]
⋮----
# Average across targets
feature_importance = np.mean(valid_importances, axis=0)
⋮----
feature_importance = None
⋮----
# If we have a multi-output target, use MultiOutputRegressor for base models
⋮----
model = MultiOutputRegressor(base_model)
⋮----
model = base_model
# Train the model with progress logging
⋮----
# Extract feature importance if available
⋮----
feature_importance = model.feature_importances_
⋮----
# Average feature importance across all estimators
importance_per_estimator = [est.feature_importances_ for est in model.estimators_]
feature_importance = np.mean(importance_per_estimator, axis=0)
⋮----
# Custom class for target-specific models
class TargetSpecificModel(BaseEstimator, RegressorMixin)
⋮----
def __init__(self, target_models=None, fallback_model='random_forest')
def fit(self, X, y)
⋮----
# Get model type for this target
model_type = self.target_models.get(col, self.fallback_model)
# Create and fit model for this target
⋮----
model = DecisionTreeRegressor(max_depth=8, min_samples_split=10, random_state=42)
⋮----
model = RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42, n_jobs=1)
⋮----
model = GradientBoostingRegressor(n_estimators=50, max_depth=4, learning_rate=0.1, random_state=42)
⋮----
model = xgb.XGBRegressor(n_estimators=50, max_depth=4, learning_rate=0.1, random_state=42, n_jobs=1)
⋮----
model = lgb.LGBMRegressor(n_estimators=50, max_depth=4, learning_rate=0.1, random_state=42, n_jobs=1)
⋮----
# Fallback to random forest
⋮----
# Fit model to this target
⋮----
def predict(self, X)
⋮----
# Get predictions for each target
predictions = []
⋮----
pred = model.predict(X)
⋮----
# Combine into a 2D array
⋮----
def get_params(self, deep=True)
def set_params(self, **parameters)
def get_feature_importance(self, target=None)
⋮----
model = self.models[target]
⋮----
# Get feature importance for all targets
importances = {}
⋮----
# Create a stacked ensemble model
def create_stacked_ensemble(X_train, y_train, low_resource_mode=False, use_linear_blend=True, time_gap=1)
⋮----
# Base estimators
estimators = []
# Add decision tree
dt_params = {
⋮----
# Add random forest with different hyperparameters than decision tree
rf_params = {
⋮----
'max_features': 'sqrt',  # Randomize features to increase diversity
⋮----
'n_jobs': 1  # Use only 1 job per estimator to avoid overloading CPU
⋮----
# Add gradient boosting
gb_params = {
⋮----
# Add XGBoost if available - with different hyperparameters from GBM
⋮----
xgb_params = {
⋮----
'learning_rate': 0.05,  # Different learning rate
⋮----
'subsample': 0.75,  # Different subsample ratio
⋮----
'n_jobs': 1  # Use only 1 job per estimator
⋮----
# Add LightGBM if available - with different hyperparameters
⋮----
lgb_params = {
⋮----
'max_depth': 5 if low_resource_mode else 6,  # Different depth
'learning_rate': 0.08,  # Different learning rate
⋮----
# Add a regularized linear model for diversity
linear_params = {
⋮----
# Final estimator (blender) - choose between simple Ridge or another model
⋮----
# Simple linear blender with ridge regression
final_estimator = Ridge(alpha=1.0)
⋮----
# More complex blender using gradient boosting
final_estimator = GradientBoostingRegressor(
⋮----
# Create the stacked model with time-series aware cross-validation
cv = 3 if low_resource_mode else 5
# Use TimeSeriesSplit for stacking to maintain chronological order
cv_splitter = TimeSeriesSplit(n_splits=cv, gap=time_gap)
stacked_model = StackingRegressor(
⋮----
cv=cv_splitter,  # Use time series split for proper validation
n_jobs=1,  # Control parallelism
passthrough=True  # Include original features alongside meta-features
⋮----
# Use TimeSeriesSplit for time-based validation with a gap
is_multioutput = isinstance(y_train, (pd.DataFrame, np.ndarray)) and y_train.shape[1] > 1
# Create a TimeSeriesSplit with a configurable gap for time-series validation
# A larger gap prevents data leakage by ensuring chronological separation between train and test folds
time_series_cv = TimeSeriesSplit(n_splits=cv, gap=gap_size)
⋮----
# Define parameter grid or space based on model type
⋮----
base_model = DecisionTreeRegressor(random_state=42)
⋮----
param_space = {
⋮----
param_grid = {
⋮----
base_model = RandomForestRegressor(random_state=42, n_jobs=1)
⋮----
base_model = GradientBoostingRegressor(random_state=42, subsample=0.8)
⋮----
base_model = xgb.XGBRegressor(random_state=42, n_jobs=1)
⋮----
base_model = lgb.LGBMRegressor(random_state=42, n_jobs=1)
⋮----
# Create a stacked ensemble with default parameters
base_model = create_stacked_ensemble(X_train, y_train, low_resource_mode=True)
# Limited parameter tuning for stacked model
⋮----
# Create model for optimization
⋮----
# Use Bayesian optimization if available and requested
⋮----
search = BayesSearchCV(
⋮----
n_iter=n_iter_bayesian,  # Configurable number of iterations
⋮----
n_jobs=1,  # Use only 1 job to avoid overloading
⋮----
n_initial_points=int(n_iter_bayesian * randomized_fraction)  # Start with random sampling
⋮----
# Use RandomizedSearchCV instead of GridSearchCV for more efficient search
if len(param_grid) > 5:  # If we have many parameters, use randomized search
# Convert param_grid to param_distributions for RandomizedSearchCV
search = RandomizedSearchCV(
⋮----
n_iter=min(20, np.prod([len(values) for values in param_grid.values()])),  # Reasonable number of iterations
⋮----
n_jobs=1,  # Use only 1 job to avoid overloading
⋮----
search = GridSearchCV(
⋮----
n_jobs=1  # Use only 1 job to avoid overloading
⋮----
# Show progress bar if available
⋮----
total_fits = cv * n_iter_bayesian  # Bayesian optimization iterations
⋮----
total_fits = cv * search.n_iter  # Randomized search iterations
⋮----
# For grid search, calculate total number of parameter combinations
total_fits = cv * np.prod([len(values) for values in param_grid.values()])
⋮----
# Define a callback for Bayesian search if available
⋮----
def on_step(optim_result)
⋮----
# Fit the search
⋮----
# Extract best parameters
best_params = search.best_params_
# Convert params from estimator__ format to direct format
⋮----
best_params_direct = {k.replace('estimator__', ''): v for k, v in best_params.items()}
⋮----
best_params_direct = best_params
⋮----
# Use column names if they're available and names aren't provided
⋮----
feature_names = X_test.columns.tolist()
⋮----
target_names = y_test.columns.tolist()
# If using time-series validation, evaluate across multiple chronological splits
⋮----
# Create a TimeSeriesSplit with a gap between train and test sets
tscv = TimeSeriesSplit(n_splits=cv, gap=time_gap)
# Sort data by index if it's DateTime indexed
⋮----
combined_data = pd.concat([X_test.reset_index(drop=True),
combined_data = combined_data.sort_index()
feature_cols = X_test.columns
X_test = combined_data[feature_cols]
y_test = combined_data.drop(columns=feature_cols)
all_y_true = []
all_y_pred = []
⋮----
X_fold = X_test.iloc[test_idx]
y_fold = y_test.iloc[test_idx]
fold_pred = model.predict(X_fold)
⋮----
y_test = pd.concat(all_y_true)
y_pred = np.vstack(all_y_pred) if len(all_y_pred[0].shape) > 1 else np.concatenate(all_y_pred)
⋮----
y_pred = pd.DataFrame(y_pred, columns=y_test.columns, index=y_test.index)
⋮----
y_pred = model.predict(X_test)
⋮----
metrics = {}
⋮----
overall_mse = mean_squared_error(y_test, y_pred)
overall_mae = mean_absolute_error(y_test, y_pred)
overall_r2 = r2_score(y_test, y_pred)
⋮----
target_y_test = y_test.iloc[:, i] if isinstance(y_test, pd.DataFrame) else y_test[:, i]
target_y_pred = y_pred.iloc[:, i] if isinstance(y_pred, pd.DataFrame) else y_pred[:, i]
mse = mean_squared_error(target_y_test, target_y_pred)
mae = mean_absolute_error(target_y_test, target_y_pred)
r2 = r2_score(target_y_test, target_y_pred)
⋮----
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
target_name = target_names[0] if target_names else 'target'
⋮----
def save_model(model, output_dir=None, model_name=None)
⋮----
output_dir = "/Users/lukesmac/Projects/nbaModel/models"
⋮----
model_name = f"nba_dt_model_{current_date}.joblib"
model_path = os.path.join(output_dir, model_name)
⋮----
# Initialize results dictionary
importance_results = {}
# Extract model-based feature importance if not using permutation_importance_only
⋮----
# If we already have feature importances from the model
⋮----
# For single output models with feature_importances_ attribute
importances = model.feature_importances_
importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
importance_df = importance_df.sort_values('importance', ascending=False)
⋮----
# For multi-output models
all_importances = []
⋮----
importances = estimator.feature_importances_
target_name = y.columns[i] if isinstance(y, pd.DataFrame) else f"target_{i}"
importance_df = pd.DataFrame({
⋮----
# Combine all target importances
combined_importance = pd.concat(all_importances)
# Get top features per target
top_features_per_target = {}
⋮----
target_imp = combined_importance[combined_importance['target'] == target]
target_imp = target_imp.sort_values('importance', ascending=False)
⋮----
# Perform permutation importance for more reliable measure
⋮----
# If low resource mode, sample a subset of data for permutation importance
⋮----
# Take a sample of 1000 rows or 20% of data, whichever is smaller
sample_size = min(1000, int(len(X) * 0.2))
sample_indices = np.random.choice(len(X), size=sample_size, replace=False)
X_sample = X.iloc[sample_indices] if isinstance(X, pd.DataFrame) else X[sample_indices]
⋮----
y_sample = y.iloc[sample_indices]
⋮----
y_sample = y[sample_indices]
⋮----
X_sample = X
y_sample = y
# If X is a DataFrame, make sure to preserve feature names
⋮----
feature_names_sample = X_sample.columns.tolist()
⋮----
feature_names_sample = feature_names
# For multi-output targets, analyze each target separately
⋮----
perm_importance_results = {}
⋮----
# Get the target column
target = y_sample[col]
# Get the corresponding estimator for multi-output models
⋮----
estimator = model.estimators_[i]
# Calculate permutation importance with reduced repeats and cores for efficiency
n_repeats = 3 if low_resource_mode else 5
⋮----
perm_importance = permutation_importance(
⋮----
# Create dataframe for this target
perm_imp_df = pd.DataFrame({
perm_imp_df = perm_imp_df.sort_values('importance_mean', ascending=False)
# Store more comprehensive results
⋮----
# Single target case with reduced repeats and cores
⋮----
# More comprehensive results for single target case
⋮----
# Save results if output directory is provided
⋮----
# Create output directory if it doesn't exist
⋮----
importance_path = os.path.join(output_dir, f"feature_importance_{current_date}.json")
⋮----
dt_variant_params = hyperparams.copy()
⋮----
rf_variant_params = hyperparams.copy()
⋮----
rf_variant2_params = hyperparams.copy()
⋮----
gb_variant_params = hyperparams.copy()
⋮----
xgb_variant_params = hyperparams.copy()
⋮----
lgb_variant_params = hyperparams.copy()
⋮----
final_estimator = RandomForestRegressor(
⋮----
def save_metrics(metrics, feature_names, target_names, output_dir=None, metrics_name=None)
⋮----
metrics_name = f"nba_dt_metrics_{current_date}.json"
metrics_path = os.path.join(output_dir, metrics_name)
# Combine metrics with feature and target information
metrics_data = {
⋮----
resource_mode = "low resource mode (reduced CPU usage)" if low_resource_mode else "standard resource mode"
⋮----
# Check if we're using the target-specific model type
using_target_specific = (model_type == 'target_specific')
⋮----
model_per_stat = {
data = load_training_data(data_path)
⋮----
target_names = y.columns.tolist()
⋮----
target_names = ['target']
⋮----
train_separate_models = False
⋮----
# Prepare CV split with time gap for data leakage prevention
⋮----
tscv = TimeSeriesSplit(n_splits=5, gap=time_gap)
# Initialize lists to store metrics
cv_scores = {
# Initialize target-specific metrics
⋮----
# Perform cross-validation with progress tracking
⋮----
# Train model with target-specific models without hyperparameter tuning in CV
⋮----
# Evaluate on the test fold
⋮----
# Calculate overall metrics
⋮----
# Store overall metrics
⋮----
# Calculate and store target-specific metrics
⋮----
y_test_target = y_test.iloc[:, i]
y_pred_target = y_pred[:, i]
mse = mean_squared_error(y_test_target, y_pred_target)
mae = mean_absolute_error(y_test_target, y_pred_target)
r2 = r2_score(y_test_target, y_pred_target)
⋮----
# Calculate average scores
avg_metrics = {}
⋮----
# Print R² scores for each target
⋮----
# Train final model on all data
# No hyperparameter tuning for target-specific model
⋮----
# Save the model and metrics
model_path = save_model(model, model_name=f"nba_target_specific_{datetime.now().strftime('%Y%m%d')}.joblib")
metrics_path = save_metrics(avg_metrics, feature_names, target_names)
# No feature importance analysis for target-specific models
# (already handled in train_model function)
⋮----
# Regular train/test split with chronological ordering
# Sort by date if available
⋮----
data = data.sort_values('game_date')
# Re-create feature matrix after sorting
⋮----
# Take the last test_size% as test data (chronological split)
split_idx = int(len(X) * (1 - test_size))
⋮----
# Train model with target-specific models
⋮----
# Evaluate the model
metrics = evaluate_model(model, X_test, y_test, feature_names, target_names)
# Save the model
⋮----
# Save the metrics
metrics_path = save_metrics(metrics, feature_names, target_names)
⋮----
# Train a stacked ensemble model
⋮----
# Perform cross-validation
⋮----
# Create and train a stacked ensemble
⋮----
model_path = save_model(model, model_name=f"nba_stacked_{datetime.now().strftime('%Y%m%d')}.joblib")
⋮----
# Train stacked ensemble model
⋮----
# Train separate models for each stat
models = {}
all_metrics = {}
feature_importances = {}
# For each target, use a different model if better performance is known
target_models = {}
# Train separate models with progress tracking
⋮----
# Extract the target column
⋮----
y_target = y[target]
⋮----
y_target = y
# Use time-based split to respect chronological order
⋮----
# Create time series cross-validation with gap
⋮----
# Initialize lists to store metrics
⋮----
# Select model type for this target if model_per_stat is provided
⋮----
target_model_type = model_per_stat[target]
⋮----
target_model_type = model_type
# Store the model type used for this target
⋮----
# Perform cross-validation
⋮----
# Tune hyperparameters if requested
⋮----
hyperparams = tune_model_hyperparameters(X_train, y_train, target_model_type, cv=3,
⋮----
# Use default hyperparameters
hyperparams = None
# Train the model
⋮----
# Evaluate on the test fold
⋮----
# Calculate metrics
⋮----
# Store metrics
⋮----
# Calculate average scores
avg_metrics = {
# Train final model on all data
⋮----
# Store the model and metrics
⋮----
# Sort by date if available
⋮----
# Re-create feature matrix after sorting
⋮----
# Re-extract the target column
⋮----
# Take the last test_size% as test data (chronological split)
⋮----
# Tune hyperparameters if requested
⋮----
hyperparams = tune_model_hyperparameters(X_train, y_train, target_model_type,
⋮----
# Use default hyperparameters
⋮----
# Train the model
⋮----
# Evaluate the model
⋮----
# Calculate metrics
⋮----
metrics = {
⋮----
# Log the R² for this target
⋮----
# Update progress
⋮----
# Calculate overall metrics (average across targets)
overall_metrics = {
# Add overall metrics
⋮----
# Save models and metrics
⋮----
model_type_used = target_models.get(target, model_type)
model_path = save_model(model, model_name=f"nba_{model_type_used}_{target}_{datetime.now().strftime('%Y%m%d')}.joblib")
⋮----
# Save all metrics
metrics_path = save_metrics(all_metrics, feature_names, target_names)
# Analyze feature importance if available
⋮----
# Create DataFrame with feature names and importance
⋮----
# Save feature importance
⋮----
# Train a single multi-output model
⋮----
# Create time series cross-validation with gap
⋮----
hyperparams = tune_model_hyperparameters(X_train, y_train, model_type, cv=3,
⋮----
model_path = save_model(model)
⋮----
# Analyze feature importance
importance_analysis = analyze_feature_importance(model, X, y, feature_names,
⋮----
# Tune hyperparameters if requested
⋮----
hyperparams = tune_model_hyperparameters(X_train, y_train, model_type,
⋮----
# Use default hyperparameters
⋮----
# Train the model
⋮----
parser = argparse.ArgumentParser(description='Train and evaluate NBA player performance prediction model')
⋮----
args = parser.parse_args()
````

## File: src/predict.py
````python
def load_model(model_path=None)
⋮----
models_dir = "/Users/lukesmac/Projects/nbaModel/models"
⋮----
model_files = []
single_files = [f for f in os.listdir(models_dir)
⋮----
model_dirs = [d for d in os.listdir(models_dir)
⋮----
model_dir_path = os.path.join(models_dir, model_dir)
⋮----
rel_path = os.path.join(os.path.relpath(root, models_dir), file)
⋮----
model_files = sorted(model_files, reverse=True)
⋮----
model_path = os.path.join(models_dir, model_files[0])
⋮----
model = joblib.load(model_path)
⋮----
def load_model_metadata(metadata_path=None)
⋮----
metrics_files = []
old_metrics = [f for f in os.listdir(models_dir)
⋮----
target_metrics = [f for f in os.listdir(models_dir)
⋮----
metrics_files = sorted(metrics_files, reverse=True)
⋮----
metadata_path = os.path.join(models_dir, metrics_files[0])
⋮----
metadata = json.load(f)
feature_names = metadata.get('feature_names', [])
target_names = metadata.get('target_names', [])
⋮----
@profile_memory
def prepare_player_data(player_id=None, player_name=None, team=None, opponent=None, use_optimized_types=True)
⋮----
processed_dir = "/Users/lukesmac/Projects/nbaModel/data/processed"
⋮----
processed_files = sorted(
⋮----
processed_path = os.path.join(processed_dir, processed_files[0])
dtype_dict = None
usecols = None
⋮----
usecols = ['playerID', 'longName', 'teamID', 'teamAbv', 'game_date', 'gameID']
⋮----
# Add prefix columns
⋮----
# Define optimized dtypes if requested
⋮----
dtype_dict = {
# Add dtype for all std_ and last5_ columns
sample_df = pd.read_csv(processed_path, nrows=1)
⋮----
# Load the processed data with optimizations
⋮----
# Only load the columns we need with optimized dtypes
processed_data = pd.read_csv(processed_path, usecols=usecols, dtype=dtype_dict)
⋮----
# Convert date to proper format more efficiently
⋮----
# Convert string columns to categorical for string columns with few unique values
⋮----
# Filter for the specific player
⋮----
player_data = processed_data[processed_data['playerID'] == player_id].copy()
⋮----
# Try to find the player by name (case-insensitive)
player_data = processed_data[processed_data['longName'].str.lower() == player_name.lower()].copy()
⋮----
# Try partial match
player_data = processed_data[processed_data['longName'].str.lower().str.contains(player_name.lower())].copy()
⋮----
# Clear processed_data from memory as we don't need it anymore
⋮----
player_data = player_data.sort_values('game_date', ascending=False)
recent_player_data = player_data.iloc[0].to_dict()
upcoming_game = {}
player_fields = ['playerID', 'longName', 'teamID', 'teamAbv']
⋮----
opp_ratings = load_opponent_strength(opponent)
⋮----
upcoming_game['opp_strength'] = 5.0  # Middle of 1-10 scale
⋮----
# Default values if opponent ratings not found
⋮----
upcoming_game['opp_Offensive_Rating'] = 110.0  # League average
upcoming_game['opp_Defensive_Rating'] = 110.0  # League average
upcoming_game['opp_strength'] = 5.0  # Middle of 1-10 scale
# Set date to today
today = datetime.now().strftime("%Y%m%d")
⋮----
# Use season-to-date stats from the most recent game
std_fields = [col for col in recent_player_data.keys() if col.startswith('std_')]
⋮----
# Use last 5 game averages from the most recent game
last5_fields = [col for col in recent_player_data.keys() if col.startswith('last5_')]
⋮----
# Copy over season average fields
season_fields = ['PTS', 'TRB', 'AST', 'STL', 'BLK', 'TOV_y', 'MP', 'Age', 'FG%', '3P%', 'FT%', 'USG%', 'WS/48', 'OBPM', 'DBPM', 'BPM']
⋮----
# Create a DataFrame with the upcoming game
upcoming_df = pd.DataFrame([upcoming_game])
# Clear player_data from memory
⋮----
# Apply data type optimization if requested
⋮----
datetime_cols = ['game_date']
upcoming_df = optimize_dataframe(upcoming_df, categorical_threshold=5,
⋮----
def predict_performance(player_data, model=None, feature_names=None, target_names=None)
⋮----
# Load model and metadata if not provided
⋮----
model = load_model()
⋮----
# Engineer features
engineered_data = engineer_all_features(player_data)
# Create simplified feature matrix - we'll handle the features manually
df = engineered_data.copy()
model_features = getattr(model, 'feature_names_in_', None)
⋮----
feature_names = model_features
X = pd.DataFrame(0, index=df.index, columns=feature_names)
missing_features = []
⋮----
X = X.fillna(0)
⋮----
y_pred = model.predict(X)
predictions = {}
⋮----
target = target_names[0]
⋮----
pred_value = float(y_pred[0])
⋮----
pred_value = float(y_pred)
⋮----
def find_next_opponent(team)
⋮----
team = team.upper()
team_mapping = {
standard_team = team_mapping.get(team, team)
schedules_dir = "/Users/lukesmac/Projects/nbaModel/data/schedules"
⋮----
schedule_pattern = os.path.join(schedules_dir, f"{standard_team}_*.csv")
schedule_files = sorted(glob.glob(schedule_pattern), reverse=True)
⋮----
schedule_pattern = os.path.join(schedules_dir, f"{alt_team}_*.csv")
⋮----
schedule_path = schedule_files[0]
⋮----
schedule = pd.read_csv(schedule_path)
today = datetime.now().date()
# Convert date column to datetime
⋮----
# Find the next game
future_games = schedule[schedule['Date'] >= today]
⋮----
next_game = future_games.iloc[0]
# Extract opponent
⋮----
opponent_name = next_game['Opponent']
# Convert full team name to abbreviation
team_name_to_abbr = {
# Determine if it's a home game
is_home = False
⋮----
is_home = bool(next_game['Home'])
⋮----
is_home = next_game['Location'] != '@'
# Get opponent abbreviation
opponent_abbr = team_name_to_abbr.get(opponent_name, opponent_name[:3].upper())
# Get game date
game_date = next_game['Date'] if isinstance(next_game['Date'], str) else next_game['Date'].strftime('%Y-%m-%d')
⋮----
def load_opponent_strength(opponent)
⋮----
# Map team abbreviations if needed
⋮----
# Try to find the latest team ratings file
standings_dir = "/Users/lukesmac/Projects/nbaModel/data/standings"
⋮----
# Find latest team ratings file
ratings_files = sorted(
⋮----
ratings_path = os.path.join(standings_dir, ratings_files[0])
⋮----
# Load ratings
ratings = pd.read_csv(ratings_path)
# Try to find the opponent team
⋮----
# Try exact match on abbreviation first
opponent_row = ratings[ratings['Team'] == opponent]
# If not found, try mapping to full name
⋮----
opponent_full = team_mapping[opponent]
opponent_row = ratings[ratings['Team'] == opponent_full]
# If still not found, try partial match
⋮----
opponent_row = ratings[ratings['Team'] == team_name]
⋮----
# Convert the row to a dictionary
opp_ratings = opponent_row.iloc[0].to_dict()
# Add a normalized strength score (1-10 scale, higher is stronger)
⋮----
# Get min and max ratings
min_rating = ratings['Net_Rating'].min()
max_rating = ratings['Net_Rating'].max()
# Normalize to 1-10 scale
net_rating = opp_ratings['Net_Rating']
normalized_rating = 1 + 9 * (net_rating - min_rating) / (max_rating - min_rating)
⋮----
def predict_player_performance(player_id=None, player_name=None, team=None, opponent=None)
⋮----
# Game environment variables
is_home_game = None
game_date = None
# If team is provided but no opponent, try to find next opponent from schedule
⋮----
opponent_info = find_next_opponent(team)
if opponent_info[0]:  # If opponent was found
⋮----
# Prepare player data
player_data = prepare_player_data(player_id, player_name, team, opponent)
# Add home game information if available
⋮----
# Extract player's recent game stats to calculate recent averages and trends
⋮----
recent_games = get_player_recent_games(player_id, player_name)
⋮----
player_data = enrich_with_recent_stats(player_data, recent_games)
⋮----
player_info = {
⋮----
predictions = predict_performance(player_data, model, feature_names, target_names)
⋮----
result = {**player_info, 'predictions': predictions}
⋮----
def format_prediction_output(predictions)
⋮----
player_name = predictions['player_name']
team = predictions['team']
opponent = predictions['opponent']
stats = predictions['predictions']
opponent_info = ""
⋮----
strength_level = "Very Strong" if opp_ratings['strength_score'] >= 8 else \
opponent_info = f" ({strength_level} opponent)"
location_info = ""
⋮----
location_info = " (Home)" if predictions['home_game'] else " (Away)"
output = f"Predicted Performance for {player_name} ({team} vs {opponent}{opponent_info}{location_info}):\n"
⋮----
stat_display = {
⋮----
@profile_memory
def predict_multiple_players(player_list, opponent=None, batch_size=10, use_optimized_types=True)
⋮----
init_mem = memory_usage_report()
⋮----
def process_player_batch(batch)
⋮----
batch_results = []
⋮----
player_name = player.get('name')
team = player.get('team')
player_opponent = opponent
⋮----
player_data = prepare_player_data(
⋮----
recent_games = get_player_recent_games(player_name=player_name)
⋮----
engineered_data = engineer_all_features(
⋮----
key_features = ['PTS', 'TRB', 'AST', 'STL', 'BLK']
⋮----
# Process players in batches
results = []
# Use batch processing
⋮----
# Split the player list into batches and process
⋮----
batch = player_list[i:i+batch_size]
⋮----
batch_results = process_player_batch(batch)
⋮----
# Force garbage collection between batches
⋮----
mem_usage = memory_usage_report()
⋮----
# For a small number of players, process all at once
results = process_player_batch(player_list)
# Clean up
⋮----
# Log final memory usage
final_mem = memory_usage_report()
⋮----
def summarize_team_predictions(predictions)
⋮----
# Total team stats
team_stats = {
# Count players with predictions
player_count = 0
⋮----
# Get team and opponent info from first player
team = predictions[0]['team'] if predictions and 'team' in predictions[0] else 'TEAM'
opponent = predictions[0]['opponent'] if predictions and 'opponent' in predictions[0] else 'OPP'
# Create team summary
summary = {
⋮----
def get_player_recent_games(player_id=None, player_name=None, num_games=10)
⋮----
# Try to find the latest game stats file
game_stats_dir = "/Users/lukesmac/Projects/nbaModel/data/playerGameStats"
⋮----
game_stats_files = sorted(
⋮----
game_stats_path = os.path.join(game_stats_dir, game_stats_files[0])
# Load the game stats
⋮----
game_stats = pd.read_csv(game_stats_path)
⋮----
player_games = game_stats[game_stats['playerID'] == player_id]
⋮----
player_games = game_stats[game_stats['longName'].str.lower() == player_name.lower()]
⋮----
player_games = game_stats[game_stats['longName'].str.lower().str.contains(player_name.lower())]
⋮----
# Extract date from gameID and sort by date
⋮----
player_games = player_games.sort_values('game_date', ascending=False)
# Get the most recent games
recent_games = player_games.head(num_games)
⋮----
def enrich_with_recent_stats(player_data, recent_games)
⋮----
# Make a copy to avoid modifying the original
df = player_data.copy()
# Ensure recent_games has the required columns
required_cols = ['pts', 'reb', 'ast', 'stl', 'blk', 'TOV_x', 'fgm', 'fga', 'fgp',
# Map column names if they're different in recent_games
col_mapping = {
mapped_cols = {}
⋮----
last3_games = recent_games.head(3)
⋮----
avg_value = last3_games[source_col].mean()
⋮----
last10_games = recent_games.head(10)
⋮----
avg_value = last10_games[source_col].mean()
⋮----
std_value = recent_games[source_col].astype(float).std()
mean_value = recent_games[source_col].astype(float).mean()
⋮----
cv = std_value / mean_value
⋮----
def format_team_summary(summary)
⋮----
team = summary['team']
opponent = summary['opponent']
player_count = summary['player_count']
stats = summary['stats']
output = f"Team Prediction Summary for {team} vs {opponent} ({player_count} players):\n"
⋮----
parser = argparse.ArgumentParser(description='Predict NBA player performance')
⋮----
args = parser.parse_args()
⋮----
prediction = predict_player_performance(
````

## File: src/run_pipeline.py
````python
TQDM_AVAILABLE = False
PipelineTracker = None
ProgressLogger = None
⋮----
PIPELINE_STAGES = [
pipeline_tracker = None
def initialize_pipeline_tracker(use_tqdm=True)
⋮----
pipeline_tracker = PipelineTracker(stages=PIPELINE_STAGES, use_tqdm=use_tqdm)
⋮----
def run_data_processing(low_memory_mode=False, cleanup=False, max_age_days=30, keep_latest=3)
⋮----
result = process_data(low_memory_mode=low_memory_mode)
⋮----
deleted_count = cleanup_all_data_directories(
⋮----
def run_feature_engineering(recache=False, remove_redundant=True, validate_derived=True)
⋮----
current_date = datetime.now().strftime("%Y%m%d")
processed_dir = "/Users/lukesmac/Projects/nbaModel/data/processed"
processed_data_path = os.path.join(processed_dir, f"processed_nba_data_{current_date}.csv")
⋮----
processed_files = sorted(
⋮----
processed_data_path = os.path.join(processed_dir, processed_files[0])
⋮----
processed_data = pd.read_csv(processed_data_path)
⋮----
processed_data = run_all_quality_checks(
⋮----
engineered_data = engineer_all_features(
output_dir = "/Users/lukesmac/Projects/nbaModel/data/engineered"
⋮----
output_path = os.path.join(output_dir, f"engineered_nba_data_{current_date}.csv")
⋮----
initial_cols = processed_data.shape[1]
final_cols = engineered_data.shape[1]
added_cols = final_cols - initial_cols
⋮----
models_dir = "/Users/lukesmac/Projects/nbaModel/models"
⋮----
model_files = sorted(
⋮----
model_file = os.path.join(models_dir, model_files[0])
⋮----
updated = update_feature_importance_from_model(
⋮----
num_features = len(updated)
⋮----
top_features = sorted(updated.items(), key=lambda x: x[1], reverse=True)[:10]
⋮----
progress_map = None
⋮----
initial_mem = memory_usage_report()
⋮----
success = False
⋮----
players = [{'name': name, 'team': team} for name in player_names]
predictions = predict_multiple_players(
⋮----
success = True
⋮----
def predict_for_player(player_name)
⋮----
prediction = predict_player_performance(
⋮----
results = progress_map(
success = any(result is not None for result in results)
⋮----
# Fall back to regular loop
⋮----
prediction = predict_for_player(player_name)
⋮----
def run_visualizations(show_plots=False)
⋮----
# Start visualizations stage
⋮----
# Try importing without the src prefix
⋮----
# Run all visualizations
⋮----
result = create_all_visualizations(show_plots=show_plots)
⋮----
# Initialize the pipeline tracker
⋮----
pipeline_tracker = initialize_pipeline_tracker()
⋮----
# Clean all output files if fresh_start is requested
⋮----
# Try importing without the src prefix
⋮----
deleted_count = clean_all_output_files(dry_run=False)
⋮----
# Step 1: Data Processing
process_success = run_data_processing(
⋮----
cleanup=False  # Don't clean up yet, do it at the end of the pipeline
⋮----
feature_success = run_feature_engineering(
⋮----
model_success = run_model_training(
⋮----
pred_success = run_predictions(
⋮----
parser = argparse.ArgumentParser(description='Run the NBA player performance prediction pipeline')
⋮----
feature_engineering_group = parser.add_argument_group('Feature Engineering Options')
⋮----
feature_importance_group = parser.add_argument_group('Feature Importance Options')
⋮----
memory_group = parser.add_argument_group('Memory Optimization Options')
⋮----
progress_group = parser.add_argument_group('Progress Tracking Options')
⋮----
args = parser.parse_args()
⋮----
final_mem = memory_usage_report()
````

## File: src/train_target_models.py
````python
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
⋮----
data_pipeline_available = True
⋮----
data_pipeline_available = False
⋮----
new_targets = ['fgm', 'fga', 'tptfgm', 'tptfga']
⋮----
use_ensemble = False
⋮----
data = load_training_data()
⋮----
# If not, we should load them from the processed data and merge them
⋮----
# Try to load the processed data which should have these columns
processed_data_path = os.path.join("data", "processed", "processed_nba_data_20250313.csv")
⋮----
processed_data = pd.read_csv(processed_data_path)
# Check if the target column exists in the processed data
⋮----
# Add the target column to our data
⋮----
# Apply data quality and feature optimization if requested
⋮----
# First try direct import (when running from src directory)
⋮----
# Fall back to src prefix import (when running from project root)
⋮----
# First validate derived features if requested
⋮----
data = validate_derived_features(data)
⋮----
# Then remove redundant features if requested
⋮----
data = detect_and_handle_redundant_features(data)
# Also remove unused features (those with very low importance)
data = detect_unused_features(data)
⋮----
# Create feature matrix manually to avoid issues with the create_feature_matrix function
# Map target name to possible column names
column_mapping = {
# Find the actual column name in the data
actual_column = None
⋮----
actual_column = col
⋮----
actual_column = target_name
⋮----
# Extract target column
y = data[[actual_column]]
# Create a list of columns to exclude from features
exclude_cols = [
⋮----
# Target and actual column
⋮----
# Statistics columns (lowercase and uppercase)
⋮----
# Metadata columns
⋮----
# Remove target columns and non-feature columns
feature_cols = [col for col in data.columns if col not in exclude_cols]
X = data[feature_cols].copy()
# Handle missing values
⋮----
if X[col].dtype.kind in 'if':  # numeric columns (integer or float)
mean_val = X[col].mean() if not X[col].isna().all() else 0
⋮----
# For non-numeric columns, fill with the most common value
most_common = X[col].mode()[0] if not X[col].isna().all() else "unknown"
⋮----
# Get feature names
feature_names = X.columns.tolist()
⋮----
# Train model
model_result = train_model(
⋮----
# Unpack the result tuple - (model, feature_importance)
⋮----
# Evaluate model
metrics = evaluate_model(
⋮----
cv=5  # Use 5-fold cross-validation
⋮----
# Get feature importance
importance = analyze_feature_importance(
# Save the model, metrics, and feature importance
date_str = datetime.now().strftime("%Y%m%d")
# Ensure models directory exists
⋮----
# Create a mapping from actual column name to standardized target name
target_mapping = {}
⋮----
# Save model
model_dir = f"models/nba_{target_name}_model_{date_str}.joblib"
⋮----
model_path = os.path.join(model_dir, f"nba_dt_model_{date_str}.joblib")
⋮----
# Save metrics
metrics_path = f"models/nba_{target_name}_metrics_{date_str}.json"
# Add target mapping to metrics if needed
⋮----
# Add feature names to metrics
⋮----
# Save metrics manually as the save_metrics function needs target_names parameter
⋮----
# Save feature importance
importance_path = f"models/feature_importance_{target_name}_{date_str}.json"
⋮----
def main()
⋮----
parser = argparse.ArgumentParser(description="Train NBA prediction models for specific targets")
⋮----
# Feature engineering options
feature_group = parser.add_argument_group('Feature Engineering Options')
⋮----
args = parser.parse_args()
# Process data if requested and available
⋮----
processed_data = process_data(run_quality_checks=True)
⋮----
engineered_data = engineer_all_features(
⋮----
# Train models for each target
results = {}
⋮----
# Print summary
⋮----
metrics = result["metrics"]
⋮----
target_metrics = metrics[target]
r2 = target_metrics.get("r2", "N/A")
mae = target_metrics.get("mae", "N/A")
rmse = target_metrics.get("rmse", "N/A")
⋮----
r2 = metrics.get("r2", "N/A")
mae = metrics.get("mae", "N/A")
rmse = metrics.get("rmse", "N/A")
if isinstance(r2, float): r2 = f"{r2:.4f}"
if isinstance(mae, float): mae = f"{mae:.4f}"
if isinstance(rmse, float): rmse = f"{rmse:.4f}"
````

## File: .repomixignore
````
# Add patterns to ignore here, one per line
# Example:
# *.log
# tmp/
*.csv
__pycache__
````

## File: fetchGameData.py
````python
REQUEST_QUEUE = queue.Queue()
MAX_REQUESTS_PER_MINUTE = 600
REQUEST_LOCK = threading.Lock()
LAST_REQUEST_TIMES = []
def rate_limited_request(url, headers, params=None)
⋮----
current_time = time.time()
LAST_REQUEST_TIMES = [t for t in LAST_REQUEST_TIMES if current_time - t < 60]
⋮----
wait_time = 60 - (current_time - LAST_REQUEST_TIMES[0])
⋮----
# Silent waiting - no print statement
⋮----
# Recalculate current time after waiting
⋮----
# Add current request timestamp
⋮----
# Make the request
⋮----
def fetch_nba_player_list()
⋮----
url = "https://tank01-fantasy-stats.p.rapidapi.com/getNBAPlayerList"
# Get API credentials from environment variables
api_key = os.getenv('RAPIDAPI_KEY')
api_host = os.getenv('RAPIDAPI_HOST')
⋮----
headers = {
⋮----
response = rate_limited_request(url, headers)
response.raise_for_status()  # Raise an exception for HTTP errors
⋮----
def process_player_data(response_data)
⋮----
player_data = []
⋮----
# Extract player data from the response
⋮----
player_info = {
⋮----
def save_to_csv(player_data, output_dir)
⋮----
# Create the output directory if it doesn't exist
⋮----
current_date = datetime.now().strftime("%Y%m%d")
output_file = os.path.join(output_dir, f"player_info_{current_date}.csv")
⋮----
fieldnames = player_data[0].keys()
writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
# Write header and data rows
⋮----
def fetch_player_game_stats(player_id, player_name, season="2025")
⋮----
url = "https://tank01-fantasy-stats.p.rapidapi.com/getNBAGamesForPlayer"
⋮----
querystring = {
⋮----
"doubleDouble": "0"  # Not filtering for double-doubles
⋮----
response = rate_limited_request(url, headers, querystring)
⋮----
response_data = response.json()
⋮----
game_stats = []
# Extract game stats from the response
⋮----
# Skip non-game entries like statusCode
⋮----
# Add game_id to the stats dictionary
⋮----
# Ensure player_id and name are included
⋮----
def worker_fetch_player_stats(player_chunk, season, result_queue, pbar)
⋮----
player_id = player['playerID']
player_name = player['name']
# Fetch game stats
game_stats = fetch_player_game_stats(player_id, player_name, season)
# Add results to queue
⋮----
# Update progress bar
⋮----
def save_unified_game_stats(stats_list, season, output_dir)
⋮----
output_file = os.path.join(output_dir, f"all_player_games_{season}_{current_date}.csv")
⋮----
fieldnames = stats_list[0].keys()
⋮----
def fetch_all_player_game_stats(player_data_file, season="2025", num_workers=3)
⋮----
# Check if file exists
⋮----
# Read player data from CSV
⋮----
player_df = pd.read_csv(player_data_file)
⋮----
# Check if playerID column exists
⋮----
# Create output directory
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "playerGameStats")
# Convert DataFrame to list of dictionaries
players = player_df.to_dict('records')
result_queue = queue.Queue()
chunk_size = max(1, len(players) // num_workers)
player_chunks = [players[i:i + chunk_size] for i in range(0, len(players), chunk_size)]
⋮----
futures = [executor.submit(worker_fetch_player_stats, chunk, season, result_queue, pbar)
⋮----
all_game_stats = []
⋮----
def main()
⋮----
response_data = fetch_nba_player_list()
⋮----
player_data = process_player_data(response_data)
⋮----
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "playerInfo")
⋮----
player_data_file = os.path.join(output_dir, f"player_info_{current_date}.csv")
````

## File: predict.py
````python
def predict_all_stats(player_name, team=None, opponent=None)
⋮----
prediction = predict_player_performance(
⋮----
output = format_prediction_output(prediction)
⋮----
def main()
⋮----
parser = argparse.ArgumentParser(description="Predict all NBA statistics for a player")
⋮----
args = parser.parse_args()
````

## File: README.md
````markdown
# NBA Player Performance Prediction Model

This repository contains a machine learning model to predict NBA player performance.

## Project Structure

- `data/`: Contains different types of data used in the model
  - `processed/`: Processed NBA data
  - `engineered/`: Engineered features for the model
  - `playerGameStats/`: Game statistics for individual players
  - `standings/`: Team standings and ratings
  - `schedules/`: Team schedules
  
- `models/`: Saved models and metrics
  - Points prediction model: nba_pts_model_*.joblib
  - Rebounds prediction model: nba_reb_model_*.joblib
  - Assists prediction model: nba_ast_model_*.joblib
  - Field Goals Made prediction model: nba_fgm_model_*.joblib
  - Field Goals Attempted prediction model: nba_fga_model_*.joblib
  - Three-Point Field Goals Made prediction model: nba_tptfgm_model_*.joblib
  - Three-Point Field Goals Attempted prediction model: nba_tptfga_model_*.joblib

- `src/`: Source code for the project
  - `data_processing.py`: Data processing utilities
  - `feature_engineering.py`: Feature engineering code
  - `model_builder.py`: Model building and evaluation
  - `predict.py`: Prediction functionality
  - `run_pipeline.py`: End-to-end pipeline for training and prediction

## Quick Start

### Training Models

Train all target models:
```
python train_all_targets.py
```

Or train models for specific targets:
```
python train_all_targets.py --targets pts reb ast
```

### Making Predictions

Make predictions for a specific player:
```
python predict.py --player "LeBron James" --team LAL --opponent GSW
```

## Model Performance

Current model performance:

| Target | R² | MAE | RMSE |
|--------|------|------|------|
| Points | 0.88 | 2.03 | 3.10 |
| Rebounds | 0.68 | 1.40 | 1.92 |
| Assists | 0.99 | 0.03 | 0.17 |
| FG Made | 0.85 | 0.80 | 1.22 |

*Note: Performance metrics updated after implementing new features (March 2025)*

## Top Features

The most important features for predicting points:
1. Points Per Shot (PPS)
2. True Shooting Percentage (TS%)
3. Last 10 games weighted points average
4. Field goal attempts consistency
5. Effective Field Goal Percentage (eFG%)

### Added Feature Categories

1. **Defensive Matchup Features**
   - Team defensive ratings
   - Position-specific defensive matchups
   - Recent defensive performance trends

2. **Team Lineup Context Features**
   - Lineup continuity measures
   - Player role consistency
   - Team chemistry metrics
   - Player tenure with team

3. **Time-Weighted Features**
   - Exponential decay weighting by recency
   - Performance momentum indicators
   - Exponential moving averages with different alpha values

## Future Improvements

✅ Incorporate defensive matchup data
✅ Add team/lineup context features
✅ Create time-weighted features for recency bias

Planned:
- Add more target statistics (steals, blocks, etc.)
- Create a web interface for predictions
- Add player comparison functionality
- Implement model ensembles for improved accuracy
- Add optimization for player prop betting
````

## File: repomix.config.json
````json
{
  "output": {
    "filePath": "codebase-summary.md",
    "style": "markdown",
    "parsableStyle": false,
    "fileSummary": true,
    "directoryStructure": true,
    "removeComments": true,
    "removeEmptyLines": true,
    "compress": true,
    "topFilesLength": 5,
    "showLineNumbers": true,
    "copyToClipboard": false
  },
  "include": [],
  "ignore": {
    "useGitignore": true,
    "useDefaultPatterns": true,
    "customPatterns": []
  },
  "security": {
    "enableSecurityCheck": true
  },
  "tokenCount": {
    "encoding": "o200k_base"
  }
}
````

## File: scrapeData.py
````python
USER_AGENTS = [
session = requests.Session()
def get_random_user_agent()
def throttled_request(url, session=None, initial_delay=1.0, base_delay=2.0, max_retries=3)
⋮----
# Set up headers to mimic a browser
headers = {
# Initial delay before first request
⋮----
# Create a progress bar for the request attempts
⋮----
# Try making the request with exponential backoff for retries
⋮----
response = session.get(url, headers=headers, timeout=30)
⋮----
# Success case
⋮----
# Handle different status codes
elif response.status_code == 404:  # Not Found - page doesn't exist
⋮----
wait_time = base_delay * (2 ** attempt)
jitter = random.uniform(0.5, 1.5)
wait_time = wait_time * jitter
⋮----
def scrape_standings(start_season=2024, end_season=2025)
⋮----
standings_data = {}
seasons = list(range(start_season, end_season + 1))
⋮----
url = f"https://www.basketball-reference.com/leagues/NBA_2025_ratings.html"
response = throttled_request(url)
⋮----
soup = BeautifulSoup(response.content, 'html.parser')
⋮----
team_ratings_table = soup.find("table", {'id': 'ratings'})
⋮----
ratings_html = StringIO(str(team_ratings_table))
ratings_df = pd.read_html(ratings_html)[0]
⋮----
ratings_df = clean_standings_data(ratings_df)
⋮----
def clean_standings_data(df)
⋮----
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
⋮----
def save_standings_to_csv(standings_data, output_dir='/Users/lukesmac/Projects/nbaModel/data')
⋮----
# Create standings directory if it doesn't exist
standings_dir = os.path.join(output_dir, 'standings')
⋮----
old_ratings_files = glob.glob(os.path.join(standings_dir, "team_ratings_*.csv"))
⋮----
old_standings_files = glob.glob(os.path.join(standings_dir, "team_standings_*.csv"))
⋮----
column_mapping = {
latest_season_df = None
latest_season = None
⋮----
df = df.rename(columns={old_col: new_col})
season_year = int(season.split('-')[0])
⋮----
latest_season = season_year
latest_season_df = df.copy()
⋮----
today_date = datetime.now().strftime('%Y%m%d')
ratings_filename = f"team_ratings_{today_date}.csv"
ratings_file_path = os.path.join(standings_dir, ratings_filename)
⋮----
def scrape_team_schedules(team_abbrs=None)
⋮----
team_abbrs = {
team_schedules = {}
⋮----
url = f"https://www.basketball-reference.com/teams/{abbr}/2025_games.html"
response = throttled_request(url, max_retries=4, base_delay=5)
⋮----
games_table = soup.find('table', {'id': 'games'})
⋮----
game_location_th = games_table.find('th', {'data-stat': 'game_location'})
game_location_exists = game_location_th is not None
⋮----
games_html = StringIO(str(games_table))
schedule_df = pd.read_html(games_html)[0]
⋮----
location_col_exists = False
⋮----
location_col_exists = True
⋮----
game_locations = []
rows = games_table.find_all('tr')
⋮----
location_cell = row.find('td', {'data-stat': 'game_location'})
⋮----
location = location_cell.text.strip()
⋮----
schedule_df = clean_schedule_data(schedule_df)
⋮----
def clean_schedule_data(df)
⋮----
df = df.dropna(subset=['Date'])
⋮----
score_available = df['Tm'].notna() & df['Opp'].notna()
⋮----
def save_schedules_to_csv(schedules, output_dir='/Users/lukesmac/Projects/nbaModel/data')
⋮----
# Today's date for filename
today_str = datetime.now().strftime('%Y%m%d')
⋮----
# Create a copy of the dataframe to avoid modifying the original
df_to_save = schedule_df.copy()
# Create output path
output_path = os.path.join(output_dir, f"{abbr}_schedule_{today_str}.csv")
# Check if this team's schedule file already exists for today and handle it
⋮----
existing_df = pd.read_csv(output_path)
⋮----
existing_dates = df_to_save['Date'].dt.date.unique() if hasattr(df_to_save['Date'], 'dt') else []
existing_df = existing_df[~pd.to_datetime(existing_df['Date']).dt.date.isin(existing_dates)]
df_to_save = pd.concat([existing_df, df_to_save], ignore_index=True)
⋮----
df_to_save = df_to_save.sort_values('Date')
⋮----
def find_latest_data_files(pattern, data_dir='/Users/lukesmac/Projects/nbaModel/data')
⋮----
search_pattern = os.path.join(data_dir, pattern)
files = glob.glob(search_pattern)
⋮----
def scrape_player_averages(start_season=2015, end_season=2025)
⋮----
player_averages = {}
⋮----
url = f"https://www.basketball-reference.com/leagues/NBA_{season}_per_game.html"
⋮----
player_stats_table = soup.find("table", {'id': 'per_game_stats'})
⋮----
stats_html = StringIO(str(player_stats_table))
stats_df = pd.read_html(stats_html)[0]
team_abbrs = []
rows = player_stats_table.find_all('tr')
⋮----
# Skip header rows
⋮----
# Find the team abbreviation cell (data-stat="team_name_abbr")
team_cell = row.find('td', {'data-stat': 'team_id'}) or row.find('td', {'data-stat': 'team_name_abbr'})
⋮----
team_abbrs.append(None)  # Add None if not found
# Add team abbreviations to dataframe if we have the right number
⋮----
# Clean the data
stats_df = clean_player_averages_data(stats_df, season)
# Add to dictionary
⋮----
# Try alternative approach - look for any table with per game stats
tables = soup.find_all("table")
found = False
⋮----
# Check if this looks like a player stats table
⋮----
stats_html = StringIO(str(table))
⋮----
# Extract team abbreviations directly from HTML
⋮----
rows = table.find_all('tr')
⋮----
# Skip header rows
⋮----
# Find the team abbreviation cell (data-stat="team_name_abbr")
⋮----
team_abbrs.append(None)  # Add None if not found
# Add team abbreviations to dataframe if we have the right number
⋮----
# Clean the data
⋮----
# Add to dictionary
⋮----
found = True
⋮----
def clean_player_averages_data(df, season)
⋮----
# Handle potential header rows - first check if 'Rk' column exists
⋮----
# Convert 'Rk' to string first to safely use str methods
⋮----
# Remove header rows that get included as data
df = df[~df['Rk'].str.contains('Rk', na=False)]
# Handle multi-level columns if present
⋮----
# Reset index after removing rows
df = df.reset_index(drop=True)
# Handle team column - prioritize our custom Team_Abbr if it exists
⋮----
# Rename to standard column name
⋮----
df = df.drop(columns=['Team_Abbr'])
⋮----
# If we have 'Team' but not 'Tm', rename it
df = df.rename(columns={'Team': 'Tm'})
# Convert numeric columns to float, but first identify non-numeric columns
non_numeric_cols = ['Player', 'Pos', 'Tm']
# Only use columns that actually exist in the dataframe
existing_non_numeric = [col for col in non_numeric_cols if col in df.columns]
# Convert numeric columns to float
numeric_cols = df.columns.difference(existing_non_numeric)
⋮----
# Add season information
⋮----
# Add timestamp for when this data was collected
⋮----
def scrape_player_per36_minutes(start_season=2015, end_season=2025)
⋮----
player_per36 = {}
⋮----
# NBA player per 36 minutes URL
url = f"https://www.basketball-reference.com/leagues/NBA_{season}_per_minute.html"
# Make the throttled request
⋮----
# Parse HTML content
⋮----
# Get Player Per 36 Minutes table - first try with the specific ID
player_stats_table = soup.find("table", {'id': 'per_minute_stats'})
⋮----
# Use StringIO to avoid FutureWarning
⋮----
# Extract team abbreviations directly from HTML since pandas read_html doesn't preserve data-stat attributes
⋮----
stats_df = clean_player_per36_data(stats_df, season)
⋮----
def clean_player_per36_data(df, season)
def scrape_player_per100_possessions(start_season=2015, end_season=2025)
⋮----
player_per100 = {}
⋮----
url = f"https://www.basketball-reference.com/leagues/NBA_{season}_per_poss.html"
⋮----
player_stats_table = soup.find("table", {'id': 'per_poss_stats'})
⋮----
stats_df = clean_player_per100_data(stats_df, season)
⋮----
# Try alternative approach - look for any table with per possession stats
⋮----
def clean_player_per100_data(df, season)
def scrape_player_advanced_stats(start_season=2015, end_season=2025)
⋮----
player_advanced = {}
⋮----
# NBA player advanced statistics URL
url = f"https://www.basketball-reference.com/leagues/NBA_{season}_advanced.html"
⋮----
# Get Player Advanced Stats table - first try with the specific ID
player_stats_table = soup.find("table", {'id': 'advanced_stats'})
⋮----
stats_df = clean_player_advanced_data(stats_df, season)
⋮----
def clean_player_advanced_data(df, season)
def normalize_stats_by_usage(player_averages)
⋮----
normalized_player_averages = {}
⋮----
normalized_df = df.copy()
⋮----
stats_to_normalize = ['PTS', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'FG', 'FGA', '3P', '3PA', 'FT', 'FTA']
⋮----
normalized_df = normalized_df.drop(columns=['USG_decimal'])
⋮----
def enhance_player_averages_with_advanced(player_averages, player_advanced)
⋮----
enhanced_player_averages = {}
⋮----
avg_df = player_averages[season]
advanced_df = player_advanced[season]
enhanced_df = avg_df.copy()
⋮----
exclude_cols = ['Rk', 'Player', 'Age', 'Tm', 'Pos', 'G', 'GS', 'MP',
advanced_stat_cols = [col for col in advanced_df.columns if col not in exclude_cols]
advanced_stats = {}
⋮----
enhanced_df = enhanced_df.drop(columns=['match_key'])
⋮----
def enhance_player_averages_with_per36(player_averages, player_per36)
⋮----
per36_df = player_per36[season]
⋮----
per36_stat_cols = [col for col in per36_df.columns if col not in exclude_cols]
per36_stats = {}
⋮----
per36_col_name = f"{stat_col}/36 mins"
⋮----
def enhance_player_averages_with_per100(player_averages, player_per100)
⋮----
per100_df = player_per100[season]
⋮----
per100_stat_cols = [col for col in per100_df.columns if col not in exclude_cols]
per100_stats = {}
⋮----
per100_col_name = f"{stat_col}/100 poss"
⋮----
def enhance_player_averages_with_all_stats(player_averages, player_per36, player_per100, player_advanced)
⋮----
enhanced_with_per36 = enhance_player_averages_with_per36(player_averages, player_per36)
enhanced_with_per100 = enhance_player_averages_with_per100(enhanced_with_per36, player_per100)
enhanced_with_advanced = enhance_player_averages_with_advanced(enhanced_with_per100, player_advanced)
fully_enhanced = normalize_stats_by_usage(enhanced_with_advanced)
⋮----
def save_player_averages_to_csv(player_averages, output_dir='/Users/lukesmac/Projects/nbaModel/data')
⋮----
player_stats_dir = os.path.join(output_dir, 'player_stats')
⋮----
output_path = os.path.join(player_stats_dir, f"player_averages_{today_str}.csv")
⋮----
seasons = list(player_averages.keys())
all_dfs = []
⋮----
all_seasons_df = pd.concat(all_dfs, ignore_index=True)
⋮----
existing_keys = set(existing_df['player_season_team'])
new_rows = all_seasons_df[~all_seasons_df['player_season_team'].isin(existing_keys)]
new_rows = new_rows.drop(columns=['player_season_team'])
⋮----
existing_df = existing_df.drop(columns=['player_season_team'])
combined_df = pd.concat([existing_df, new_rows], ignore_index=True)
⋮----
parser = argparse.ArgumentParser(description='Scrape NBA data from Basketball Reference')
⋮----
args = parser.parse_args()
⋮----
# Log script start
⋮----
# Print minimal info to console
⋮----
# Create a list of tasks to run based on arguments
tasks = []
⋮----
# If no specific tasks are selected, default to all
⋮----
tasks = ["standings", "player_stats"]
# Use tqdm for overall progress
⋮----
# Use tqdm.write to avoid interfering with progress bars
⋮----
standings_data = scrape_standings(start_season=args.start_season, end_season=args.end_season)
⋮----
player_stats_steps = ["averages", "per36", "per100", "advanced", "processing"]
⋮----
player_averages = scrape_player_averages(start_season=args.start_season, end_season=args.end_season)
⋮----
player_per36 = scrape_player_per36_minutes(start_season=args.start_season, end_season=args.end_season)
⋮----
player_per100 = scrape_player_per100_possessions(start_season=args.start_season, end_season=args.end_season)
⋮----
player_advanced = scrape_player_advanced_stats(start_season=args.start_season, end_season=args.end_season)
⋮----
enhanced_player_averages = enhance_player_averages_with_all_stats(
⋮----
output_path = save_player_averages_to_csv(enhanced_player_averages, args.output_dir)
````

## File: train_all_targets.py
````python
TARGET_VARIABLES = ['pts', 'reb', 'ast', 'fgm', 'fga', 'tptfgm', 'tptfga']
def train_all_models(use_ensemble=False, use_time_series=True, process_data=True, sequential=False)
⋮----
start_time = datetime.now()
⋮----
success = True
⋮----
cmd = [sys.executable, "-m", "src.train_target_models"]
⋮----
result = subprocess.run(cmd, check=True, capture_output=True, text=True)
⋮----
success = False
end_time = datetime.now()
duration = end_time - start_time
⋮----
parser = argparse.ArgumentParser(description="Train NBA prediction models for all target variables")
⋮----
process_group = parser.add_mutually_exclusive_group()
⋮----
args = parser.parse_args()
⋮----
process_data = True
⋮----
process_data = False
⋮----
success = train_all_models(
````
