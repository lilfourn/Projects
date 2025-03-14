"""
NBA Data Quality Checks for Derived Features

This module provides functions for checking and fixing quality issues in derived features
and calculated statistics in NBA player performance data.
"""

import pandas as pd
import numpy as np
import re
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def check_derived_stats(df, copy=True):
    """
    Check derived and calculated statistics for anomalies
    
    Args:
        df (pd.DataFrame): DataFrame to check
        copy (bool): Whether to make a copy of the dataframe
        
    Returns:
        pd.DataFrame: DataFrame with fixed derived stats
    """
    if copy:
        df = df.copy()
    
    # Define checks for various derived statistics
    derived_stat_checks = [
        # Check per-36 minute stats
        {
            'pattern': '_per36$',
            'min_val': 0,
            'max_val': None,  # Will be handled individually
            'specific_limits': {
                'pts_per36': 60,
                'reb_per36': 40,
                'ast_per36': 30,
                'stl_per36': 15,
                'blk_per36': 15,
                'TOV_per36': 20,
                'TOV_x_per36': 20
            }
        },
        # Check rates and efficiency metrics
        {
            'pattern': '(fgp|tptfgp|ftp|FG%|3P%|FT%|eFG%|TS%)',
            'min_val': 0,
            'max_val': 1.0,
            'specific_limits': {}
        },
        # Check other efficiency metrics
        {
            'pattern': '^PPS$',  # Points per shot
            'min_val': 0,
            'max_val': 3.0,
            'specific_limits': {}
        },
        # Check consistency features
        {
            'pattern': '_consistency$',
            'min_val': 0,
            'max_val': 1.0,
            'specific_limits': {}
        },
        # Check ratio metrics 
        {
            'pattern': '(AST_TO_ratio)',
            'min_val': 0,
            'max_val': 20.0,
            'specific_limits': {}
        }
    ]
    
    # Track issues found
    issues_found = 0
    
    # Apply checks
    for check in derived_stat_checks:
        pattern = re.compile(check['pattern'])
        matching_cols = [col for col in df.columns if pattern.search(col)]
        
        if not matching_cols:
            continue
            
        for col in matching_cols:
            # Get limits for this column
            min_val = check['min_val']
            
            # Get specific max value if available, otherwise use default
            if col in check['specific_limits']:
                max_val = check['specific_limits'][col]
            else:
                max_val = check['max_val']
            
            # Check for below minimum
            if min_val is not None:
                below_min = (df[col] < min_val).sum()
                if below_min > 0:
                    logging.warning(f"Found {below_min} values below minimum ({min_val}) in {col}")
                    df[col] = df[col].clip(lower=min_val)
                    issues_found += below_min
            
            # Check for above maximum
            if max_val is not None:
                above_max = (df[col] > max_val).sum()
                if above_max > 0:
                    logging.warning(f"Found {above_max} values above maximum ({max_val}) in {col}")
                    df[col] = df[col].clip(upper=max_val)
                    issues_found += above_max
    
    # Check for negative minutes
    if 'mins' in df.columns:
        neg_mins = (df['mins'] < 0).sum()
        if neg_mins > 0:
            logging.warning(f"Found {neg_mins} records with negative minutes, fixing")
            df['mins'] = df['mins'].clip(lower=0)
            issues_found += neg_mins
    
    # Check for suspicious values in trend features
    trend_cols = [col for col in df.columns if col.endswith('_trend') or col.endswith('_w_trend')]
    for col in trend_cols:
        # Get the base stat from the trend column
        base_stat = col.replace('_trend', '').replace('_w', '')
        
        # Skip if base stat doesn't exist
        if base_stat not in df.columns:
            continue
            
        # Calculate reasonable limits based on the base stat's distribution
        base_std = df[base_stat].std()
        if pd.notna(base_std) and base_std > 0:
            # Flag trends more than 5 standard deviations from the mean
            min_val = -5 * base_std
            max_val = 5 * base_std
            
            # Count out-of-range values
            below_min = (df[col] < min_val).sum()
            above_max = (df[col] > max_val).sum()
            
            # Fix if needed
            if below_min > 0 or above_max > 0:
                total_issues = below_min + above_max
                logging.warning(f"Found {total_issues} extreme values in {col}, capping to ±5 standard deviations")
                df[col] = df[col].clip(lower=min_val, upper=max_val)
                issues_found += total_issues
    
    # Log summary
    if issues_found > 0:
        logging.info(f"Fixed {issues_found} anomalies in derived statistics")
    else:
        logging.info("No anomalies found in derived statistics")
    
    return df

def check_and_fix_weighted_averages(df, copy=True):
    """
    Check weighted averages for consistency issues
    
    Args:
        df (pd.DataFrame): DataFrame containing weighted averages
        copy (bool): Whether to make a copy of the dataframe
        
    Returns:
        pd.DataFrame: DataFrame with fixed weighted averages
    """
    if copy:
        df = df.copy()
    
    # Find all weighted average columns
    w_avg_cols = [col for col in df.columns if col.endswith('_w_avg')]
    
    # If no weighted average columns, return original dataframe
    if not w_avg_cols:
        return df
    
    # Track issues found
    issues_fixed = 0
    
    # Process each weighted average column
    for col in w_avg_cols:
        # Identify the corresponding regular average column
        reg_col = col.replace('_w_avg', '_avg')
        
        # Check if regular column exists
        if reg_col not in df.columns:
            continue
        
        # Calculate correlation between weighted and regular average
        try:
            correlation = df[col].corr(df[reg_col])
            
            # If correlation is very low, there might be issues
            if correlation < 0.5:
                logging.warning(f"Low correlation ({correlation:.2f}) between {col} and {reg_col}")
                
                # Check for extreme differences
                diff = (df[col] - df[reg_col]).abs()
                extreme_diff_ratio = (diff > 3 * diff.std()).mean()
                
                if extreme_diff_ratio > 0.01:  # More than 1% of extreme differences
                    # Fix extreme differences by capping
                    threshold = 3 * diff.std()
                    extreme_count = (diff > threshold).sum()
                    
                    # Identify extreme rows
                    extreme_rows = diff > threshold
                    
                    # For these rows, replace weighted avg with regular avg
                    df.loc[extreme_rows, col] = df.loc[extreme_rows, reg_col]
                    
                    logging.warning(f"Fixed {extreme_count} extreme differences between {col} and {reg_col}")
                    issues_fixed += extreme_count
        except Exception as e:
            logging.warning(f"Error checking weighted averages for {col}: {str(e)}")
    
    # Log summary
    if issues_fixed > 0:
        logging.info(f"Fixed {issues_fixed} issues in weighted averages")
    
    return df


def run_derived_checks(df, copy=True):
    """
    Run all checks for derived statistics
    
    Args:
        df (pd.DataFrame): DataFrame to check
        copy (bool): Whether to make a copy of the dataframe
        
    Returns:
        pd.DataFrame: DataFrame with fixed derived statistics
    """
    if copy:
        df = df.copy()
    
    # Check basic derived stats first
    df = check_derived_stats(df, copy=False)
    
    # Then check weighted averages
    df = check_and_fix_weighted_averages(df, copy=False)
    
    return df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Check derived stats in NBA data")
    parser.add_argument("--input-file", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--output-file", type=str, help="Path to save fixed CSV file (default: add _fixed suffix)")
    
    args = parser.parse_args()
    
    # Load data
    logging.info(f"Loading data from {args.input_file}")
    try:
        df = pd.read_csv(args.input_file)
        logging.info(f"Loaded data with {df.shape[0]} rows and {df.shape[1]} columns")
    except Exception as e:
        logging.error(f"Error loading data: {str(e)}")
        exit(1)
    
    # Run checks
    fixed_df = run_derived_checks(df)
    
    # Determine output path
    if args.output_file:
        output_path = args.output_file
    else:
        base_path = args.input_file.rsplit(".", 1)[0]
        output_path = f"{base_path}_fixed.csv"
    
    # Save fixed data
    fixed_df.to_csv(output_path, index=False)
    logging.info(f"Saved fixed data to {output_path}")