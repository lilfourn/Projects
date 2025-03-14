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