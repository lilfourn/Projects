# NBA Model - Project Structure Reorganization

This document outlines the new directory structure for the NBA Model project and provides guidance on how to migrate to and use the new structure.

## New Directory Structure

The project has been reorganized into a more logical structure with clear separation of concerns:

```
nbaModel/
├── data/                    # Data directory (unchanged)
│   ├── projections/         # Projection data
│   ├── processed/           # Processed data
│   └── ...
├── src/                     # Source code (reorganized)
│   ├── __init__.py          # Package initialization
│   ├── data_collection/     # Data collection scripts
│   │   ├── __init__.py
│   │   ├── fetchGameData.py # Fetch NBA player game statistics
│   │   ├── getProjections.py # Fetch NBA projections from PrizePicks API
│   │   └── scrapeData.py    # Scrape NBA standings, player stats
│   ├── data_processing/     # Data processing scripts
│   │   ├── __init__.py
│   │   ├── data_cleanup.py  # Clean up data files
│   │   ├── data_processing.py # Process raw data
│   │   ├── data_quality.py  # Perform data quality checks
│   │   ├── data_quality_check_derived.py # Check derived features
│   │   ├── enhanced_defensive_features.py # Defensive features
│   │   └── feature_engineering.py # Create features for modeling
│   ├── models/              # Model scripts
│   │   ├── __init__.py
│   │   ├── ensemble_models.py # Ensemble model implementations
│   │   ├── model_builder.py # Build ML models
│   │   ├── predict.py       # Make predictions using trained models
│   │   ├── train_all_targets.py # Train models for all targets
│   │   ├── train_target_models.py # Train models for specific targets
│   │   └── train_xgboost_models.py # Train XGBoost models
│   ├── utils/               # Utility scripts
│   │   ├── __init__.py
│   │   ├── config.py        # Configuration settings
│   │   ├── memory_utils.py  # Memory optimization utilities
│   │   └── run_pipeline.py  # Orchestrate the entire pipeline
│   └── visualization/       # Visualization scripts
│       ├── __init__.py
│       ├── analyze_projections.py # Analyze projection data
│       ├── feature_importance_viz.py # Visualize feature importance
│       ├── feature_viz.py   # Visualize features
│       └── model_comparison.py # Compare different models
├── main.py                  # Main entry point for the application
├── migrate_structure.py     # Script to migrate to the new structure
├── setup.py                 # Package setup script
└── *_wrapper.py            # Wrapper scripts for backward compatibility
```

## Migration Process

To migrate to the new structure, follow these steps:

1. Run the migration script:
   ```bash
   python3 migrate_structure.py
   ```
   
   This script will:
   - Create the new directory structure
   - Copy files to their new locations
   - Create wrapper files for backward compatibility
   - Update import statements in the copied files

2. Test the new structure:
   ```bash
   # Run the main entry point
   python3 main.py fetch_game_data
   
   # Or use the wrapper scripts
   python3 fetchGameData_wrapper.py
   ```

3. Install the package in development mode:
   ```bash
   pip3 install -e .
   ```
   
   This will make the command-line tools available:
   ```bash
   nba-fetch-game-data
   nba-scrape-data
   nba-get-projections
   nba-process-data
   nba-train-models
   nba-train-xgboost
   nba-predict
   nba-pipeline
   ```

## Using the New Structure

### As a Python Package

```python
# Import modules from the package
from nbamodel.data_collection import fetchGameData, getProjections, scrapeData
from nbamodel.models import predict, train_all_targets
from nbamodel.utils import run_pipeline

# Run the pipeline
run_pipeline.run_full_pipeline(model_type='random_forest')

# Fetch game data
fetchGameData.fetch_all_player_game_stats()

# Get projections
getProjections.fetch_and_save_projections(output_dir='data/projections')
```

### From the Command Line

```bash
# Using the main entry point
python3 main.py fetch_game_data
python3 main.py scrape_data --all
python3 main.py get_projections
python3 main.py process_data
python3 main.py train_models
python3 main.py predict
python3 main.py run_pipeline

# Using the installed command-line tools
nba-fetch-game-data
nba-scrape-data --all
nba-get-projections
nba-process-data
nba-train-models
nba-predict
nba-pipeline
```

## Benefits of the New Structure

1. **Improved Organization**: Clear separation of concerns with modules for data collection, processing, modeling, and visualization.
2. **Better Maintainability**: Related code is grouped together, making it easier to understand and maintain.
3. **Enhanced Reusability**: Code can be imported as modules, promoting reuse.
4. **Simplified Usage**: Consistent command-line interface and Python API.
5. **Backward Compatibility**: Wrapper scripts ensure existing code continues to work.

## Next Steps

1. Review and test the migrated code to ensure everything works as expected.
2. Update any external scripts or notebooks that import from the old structure.
3. Once everything is working, you can remove the original files and wrapper scripts.
4. Update documentation to reflect the new structure.
