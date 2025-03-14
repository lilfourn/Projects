#!/usr/bin/env python3
"""
Script to predict all NBA statistics for a specific player
"""

import os
import sys
import logging
import argparse

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Import the prediction functions
try:
    from src.predict import predict_player_performance, format_prediction_output
except ImportError:
    logging.error("Could not import prediction functions from src.predict")
    sys.exit(1)

def predict_all_stats(player_name, team=None, opponent=None):
    """Predict all statistics for a player"""
    # Make a base prediction
    logging.info(f"Predicting statistics for {player_name}")
    
    # Get base prediction with standard model
    prediction = predict_player_performance(
        player_name=player_name,
        team=team,
        opponent=opponent
    )
    
    if not prediction:
        logging.error(f"Failed to make predictions for {player_name}")
        return
    
    # Add other statistics if not already present
    if 'predictions' not in prediction:
        logging.error("Predictions object missing predictions field")
        return
    
    # Display the results
    output = format_prediction_output(prediction)
    print(output)

def main():
    """Main function to parse arguments and make predictions"""
    parser = argparse.ArgumentParser(description="Predict all NBA statistics for a player")
    
    parser.add_argument("--player", type=str, required=True,
                        help="Player name")
    parser.add_argument("--team", type=str,
                        help="Player's team abbreviation")
    parser.add_argument("--opponent", type=str,
                        help="Opponent team abbreviation")
    
    args = parser.parse_args()
    
    # Make predictions
    predict_all_stats(
        player_name=args.player,
        team=args.team,
        opponent=args.opponent
    )

if __name__ == "__main__":
    main()