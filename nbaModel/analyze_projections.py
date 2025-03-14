#!/usr/bin/env python3
"""
Script to analyze the PrizePicks projections CSV file and identify unnecessary columns.
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

# Path to the unified CSV file
csv_path = 'data/projections/prizepicks_unified_2025-03-14.csv'

# Load the data
print(f"Loading data from {csv_path}...")
df = pd.read_csv(csv_path)

# Basic info
print(f"\nDataset shape: {df.shape}")
print(f"Total columns: {len(df.columns)}")

# Analyze column null values
null_counts = df.isnull().sum()
null_percent = (null_counts / len(df)) * 100
null_info = pd.DataFrame({
    'null_count': null_counts,
    'null_percent': null_percent
})
high_null_cols = null_info[null_info['null_percent'] > 90].sort_values('null_percent', ascending=False)

print(f"\nColumns with >90% null values ({len(high_null_cols)} columns):")
print(high_null_cols.head(20))

# Analyze duplicate columns (columns with the same values)
print("\nChecking for duplicate columns...")
duplicate_cols = []
for i, col1 in enumerate(df.columns):
    for col2 in df.columns[i+1:]:
        if df[col1].equals(df[col2]):
            duplicate_cols.append((col1, col2))

print(f"Found {len(duplicate_cols)} duplicate column pairs:")
for col1, col2 in duplicate_cols[:10]:  # Show first 10
    print(f"  {col1} == {col2}")

# Analyze low-information columns (columns with only one unique value)
single_value_cols = []
for col in df.columns:
    if df[col].nunique() == 1:
        single_value_cols.append((col, df[col].iloc[0]))

print(f"\nColumns with only one unique value ({len(single_value_cols)} columns):")
for col, val in single_value_cols[:20]:  # Show first 20
    print(f"  {col}: {val}")

# Analyze type columns (often redundant)
type_cols = [col for col in df.columns if col.endswith('type')]
print(f"\nType columns ({len(type_cols)} columns):")
for col in type_cols:
    print(f"  {col}: {df[col].nunique()} unique values")

# Analyze ID columns (often redundant)
id_cols = [col for col in df.columns if col.endswith('id') or col == 'id']
print(f"\nID columns ({len(id_cols)} columns):")
for col in id_cols:
    print(f"  {col}: {df[col].nunique()} unique values")

# Suggest columns to keep based on analysis
print("\nSuggested columns to keep:")

# Identify key projection attributes
projection_attrs = [col for col in df.columns if col.startswith('attributes.') and null_info.loc[col, 'null_percent'] < 50]
print(f"\nKey projection attributes ({len(projection_attrs)} columns):")
for col in projection_attrs:
    print(f"  {col}: {df[col].nunique()} unique values")

# Identify key player attributes
player_attrs = [col for col in df.columns if col.startswith('attributes_player.') and null_info.loc[col, 'null_percent'] < 50]
print(f"\nKey player attributes ({len(player_attrs)} columns):")
for col in player_attrs:
    print(f"  {col}: {df[col].nunique()} unique values")

# Identify key game attributes
game_attrs = [col for col in df.columns if col.startswith('attributes_game.') and null_info.loc[col, 'null_percent'] < 50]
print(f"\nKey game attributes ({len(game_attrs)} columns):")
for col in game_attrs:
    print(f"  {col}: {df[col].nunique()} unique values")

# Create list of recommended columns to keep
essential_cols = []

# Add essential ID columns
essential_cols.append('id')  # Main projection ID
if 'id_player' in df.columns:
    essential_cols.append('id_player')  # Player ID
if 'id_game' in df.columns:
    essential_cols.append('id_game')  # Game ID

# Add essential projection attributes
essential_projection_attrs = [
    'attributes.line_score', 
    'attributes.projection_type',
    'attributes.stat_type',
    'attributes.description',
    'attributes.start_time',
    'attributes.updated_at'
]
for col in essential_projection_attrs:
    if col in df.columns:
        essential_cols.append(col)

# Add essential player attributes
essential_player_attrs = [
    'attributes_player.first_name',
    'attributes_player.last_name',
    'attributes_player.position',
    'attributes_player.team_name'
]
for col in essential_player_attrs:
    if col in df.columns:
        essential_cols.append(col)

# Add essential game attributes
essential_game_attrs = [
    'attributes_game.away_team_id',
    'attributes_game.home_team_id',
    'attributes_game.status',
    'attributes_game.start_time'
]
for col in essential_game_attrs:
    if col in df.columns:
        essential_cols.append(col)

# Add stat average attributes if available
if 'attributes_stat_avg.average' in df.columns:
    essential_cols.append('attributes_stat_avg.average')

# Add stat type attributes if available
if 'attributes_stat_type.name' in df.columns:
    essential_cols.append('attributes_stat_type.name')

print(f"\nRecommended columns to keep ({len(essential_cols)} columns):")
for col in essential_cols:
    if col in df.columns:
        print(f"  {col}")
    else:
        print(f"  {col} (not found in dataset)")

# Create a sample of the recommended dataset
if all(col in df.columns for col in essential_cols):
    sample_df = df[essential_cols].head(5)
    print("\nSample of recommended dataset (first 5 rows):")
    print(sample_df)
