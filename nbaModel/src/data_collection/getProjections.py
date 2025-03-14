#!/usr/bin/env python3
"""
Script to fetch NBA projections from PrizePicks API and store them in CSV format.
"""

import os
import sys
import requests
import pandas as pd
import json
from datetime import datetime
import argparse
from tqdm import tqdm

def call_endpoint(url, max_level=3, include_new_player_attributes=False):
    '''
    takes: 
        - url (str): the API endpoint to call
        - max_level (int): level of json normalizing to apply
        - include_player_attributes (bool): whether to include player object attributes in the returned dataframe
    returns:
        - df (pd.DataFrame): a dataframe of the call response content
    '''
    resp = requests.get(url).json()
    
    # Check if data is available
    if not resp.get('data'):
        print(f"No data available from endpoint: {url}")
        return pd.DataFrame(), resp
    
    data = pd.json_normalize(resp['data'], max_level=max_level)
    
    # Check if included data is available
    if not resp.get('included'):
        print("No included data available")
        return data, resp
    
    included = pd.json_normalize(resp['included'], max_level=max_level)
    
    if include_new_player_attributes:
        inc_cop = included[included['type'] == 'new_player'].copy().dropna(axis=1)
        if not inc_cop.empty:
            data = pd.merge(data
                           , inc_cop
                           , how='left'
                           , left_on=['relationships.new_player.data.id'
                                     ,'relationships.new_player.data.type']
                           , right_on=['id', 'type']
                           , suffixes=('', '_new_player'))
    
    return data, resp

def fetch_and_save_projections(output_dir, include_player_attributes=True, unified=True):
    """
    Fetch projections from PrizePicks API and save to CSV.
    
    Args:
        output_dir (str): Directory to save CSV files
        include_player_attributes (bool): Whether to include player attributes
        unified (bool): Whether to create a unified CSV with all data
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get current date for filename
    current_date = datetime.now().strftime('%Y-%m-%d')
    
    # PrizePicks API URL for NBA (league_id=7)
    url = 'https://partner-api.prizepicks.com/projections?league_id=7&per_page=1000'
    
    print(f"Fetching projections from PrizePicks API...")
    df, resp = call_endpoint(url, max_level=3, include_new_player_attributes=include_player_attributes)
    
    if df.empty:
        print("No projections data available.")
        return
    
    # Create a unified dataframe with all related data
    if unified:
        print("Creating unified projections dataset...")
        
        # Start with the main projections data
        unified_df = df.copy()
        
        # Process included data
        if 'included' in resp and resp['included']:
            included_df = pd.json_normalize(resp['included'], max_level=3)
            
            # Create dictionary of dataframes by type for easier joining
            type_dfs = {}
            for data_type in included_df['type'].unique():
                type_dfs[data_type] = included_df[included_df['type'] == data_type].copy()
            
            # Join player data
            if 'new_player' in type_dfs:
                player_df = type_dfs['new_player']
                unified_df = pd.merge(
                    unified_df,
                    player_df,
                    how='left',
                    left_on=['relationships.new_player.data.id'],
                    right_on=['id'],
                    suffixes=('', '_player')
                )
            
            # Join team data
            if 'team' in type_dfs and 'relationships.team.data.id' in unified_df.columns:
                team_df = type_dfs['team']
                unified_df = pd.merge(
                    unified_df,
                    team_df,
                    how='left',
                    left_on=['relationships.team.data.id'],
                    right_on=['id'],
                    suffixes=('', '_team')
                )
            
            # Join game data
            if 'game' in type_dfs:
                game_df = type_dfs['game']
                # Extract game IDs from the relationships
                if 'relationships.game.data.id' in unified_df.columns:
                    unified_df = pd.merge(
                        unified_df,
                        game_df,
                        how='left',
                        left_on=['relationships.game.data.id'],
                        right_on=['id'],
                        suffixes=('', '_game')
                    )
            
            # Join stat_average data
            if 'stat_average' in type_dfs:
                stat_avg_df = type_dfs['stat_average']
                unified_df = pd.merge(
                    unified_df,
                    stat_avg_df,
                    how='left',
                    left_on=['relationships.stat_average.data.id'],
                    right_on=['id'],
                    suffixes=('', '_stat_avg')
                )
            
            # Join stat_type data
            if 'stat_type' in type_dfs:
                stat_type_df = type_dfs['stat_type']
                unified_df = pd.merge(
                    unified_df,
                    stat_type_df,
                    how='left',
                    left_on=['relationships.stat_type.data.id'],
                    right_on=['id'],
                    suffixes=('', '_stat_type')
                )
        
        # Extract home and away teams from game data
        print("Extracting home and away teams from game data...")
        
        # Create a mapping of game IDs to home/away teams
        game_team_mapping = {}
        
        # Process the included data to extract game team information
        if 'included' in resp and resp['included']:
            for item in resp['included']:
                if item['type'] == 'game':
                    game_id = item['id']
                    if 'attributes' in item and 'metadata' in item['attributes'] and 'game_info' in item['attributes']['metadata']:
                        game_info = item['attributes']['metadata']['game_info']
                        if 'teams' in game_info:
                            teams = game_info['teams']
                            home_team = teams.get('home', {}).get('abbreviation', '')
                            away_team = teams.get('away', {}).get('abbreviation', '')
                            game_team_mapping[game_id] = (home_team, away_team)
        
        # Filter to keep only essential columns
        essential_columns = [
            # Projection attributes
            'attributes.line_score',
            'attributes.stat_display_name',
            'attributes.description',
            
            # Player attributes
            'attributes.name',
            'attributes.position',
            'attributes.team',
            
            # Game ID for joining with team info
            'relationships.game.data.id',
        ]
        
        # Only keep columns that exist in the dataframe
        existing_columns = [col for col in essential_columns if col in unified_df.columns]
        
        # Create a clean, filtered dataframe
        filtered_df = unified_df[existing_columns].copy()
        
        # Rename columns for clarity
        column_mapping = {
            'attributes.line_score': 'line_score',
            'attributes.stat_display_name': 'stat_type',
            'attributes.description': 'team',
            'attributes.name': 'player_name',
            'attributes.position': 'position',
            'attributes.team': 'player_team',
            'relationships.game.data.id': 'game_id',
        }
        
        # Only use column names that exist in the filtered dataframe
        rename_mapping = {k: v for k, v in column_mapping.items() if k in existing_columns}
        filtered_df = filtered_df.rename(columns=rename_mapping)
        
        # Add home and away team columns based on game_id
        if 'game_id' in filtered_df.columns:
            # Function to get home and away teams from game_id
            def get_teams(game_id):
                if game_id in game_team_mapping:
                    return pd.Series(game_team_mapping[game_id])
                return pd.Series(['', ''])
            
            # Apply the function to create home_team and away_team columns
            filtered_df[['home_team', 'away_team']] = filtered_df['game_id'].apply(get_teams)
            
            # Drop the game_id column as it's no longer needed
            filtered_df = filtered_df.drop(columns=['game_id'])
        
        # Filter out projections with "Combo" in stat_type
        print("Filtering out 'Combo' projections...")
        initial_count = len(filtered_df)
        filtered_df = filtered_df[~filtered_df['stat_type'].str.contains('Combo', case=False, na=False)]
        combo_count = initial_count - len(filtered_df)
        print(f"Removed {combo_count} 'Combo' projections")
        
        # Save unified dataset
        unified_file = os.path.join(output_dir, f'prizepicks_unified_{current_date}.csv')
        filtered_df.to_csv(unified_file, index=False)
        print(f"Saved optimized dataset with {len(filtered_df)} records and {len(filtered_df.columns)} columns to {unified_file}")
        
        # Print column names for reference
        print(f"Columns in the optimized dataset: {', '.join(filtered_df.columns.tolist())}")
    
    # Save individual files if requested
    if not unified:
        # Save main projections
        output_file = os.path.join(output_dir, f'prizepicks_projections_{current_date}.csv')
        df.to_csv(output_file, index=False)
        print(f"Saved {len(df)} projections to {output_file}")
        
        # Save included data separately
        try:
            if 'included' in resp and resp['included']:
                included_df = pd.json_normalize(resp['included'], max_level=3)
                
                # Split included data by type
                for data_type in included_df['type'].unique():
                    type_df = included_df[included_df['type'] == data_type]
                    type_file = os.path.join(output_dir, f'prizepicks_{data_type}_{current_date}.csv')
                    type_df.to_csv(type_file, index=False)
                    print(f"Saved {len(type_df)} {data_type} records to {type_file}")
        except Exception as e:
            print(f"Error saving included data: {e}")

def main():
    # Get the project root directory
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    default_output = os.path.join(project_root, 'data/projections')
    
    parser = argparse.ArgumentParser(description='Fetch NBA projections from PrizePicks API')
    parser.add_argument('--output', '-o', default=default_output, 
                        help='Directory to save projection files')
    parser.add_argument('--include-player', '-p', action='store_true', default=True,
                        help='Include player attributes in the projections')
    parser.add_argument('--unified', '-u', action='store_true', default=True,
                        help='Create a unified CSV with all data (default: True)')
    parser.add_argument('--separate', '-s', action='store_true', default=False,
                        help='Create separate CSV files for each data type')
    
    args = parser.parse_args()
    
    # If --separate is specified, override --unified
    unified = args.unified
    if args.separate:
        unified = False
    
    fetch_and_save_projections(args.output, args.include_player, unified)

if __name__ == '__main__':
    main()