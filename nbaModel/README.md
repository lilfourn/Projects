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