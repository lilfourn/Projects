#!/usr/bin/env python3
"""
Setup script for the NBA Model package.
"""

from setuptools import setup, find_packages

setup(
    name="nbamodel",
    version="1.0.0",
    description="NBA Player Performance Prediction Model",
    author="NBA Model Team",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.4.0",
        "requests>=2.25.0",
        "beautifulsoup4>=4.9.0",
        "tqdm>=4.60.0",
        "python-dotenv>=0.19.0",
        "xgboost>=1.5.0",
    ],
    entry_points={
        "console_scripts": [
            "nba-fetch-game-data=nbamodel.data_collection.fetchGameData:main",
            "nba-scrape-data=nbamodel.data_collection.scrapeData:main",
            "nba-get-projections=nbamodel.data_collection.getProjections:main",
            "nba-process-data=nbamodel.data_processing.data_processing:main",
            "nba-train-models=nbamodel.models.train_all_targets:main",
            "nba-train-xgboost=nbamodel.models.train_xgboost_models:main",
            "nba-predict=nbamodel.models.predict:main",
            "nba-pipeline=nbamodel.utils.run_pipeline:run_full_pipeline",
        ],
    },
)
