#!/usr/bin/env python3
# Memory Optimization and Progress Tracking Utilities for NBA Model
# This module provides tools for memory optimization, profiling, and progress tracking

import os
import gc
import logging
import psutil
import numpy as np
import pandas as pd
from functools import wraps
from typing import Callable, List, Any, Optional, Union, Dict, Iterable
import time

# Try to import tqdm for progress bars
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    logging.warning("tqdm not available for progress bars. Install with 'pip install tqdm'")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Memory tracking functions

def get_memory_usage():
    """
    Get current memory usage of the process
    
    Returns:
        float: Memory usage in MB
    """
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / (1024 * 1024)  # Convert to MB

def memory_usage_report():
    """
    Log current memory usage
    
    Returns:
        float: Memory usage in MB
    """
    mem_usage = get_memory_usage()
    logging.info(f"Current memory usage: {mem_usage:.2f} MB")
    return mem_usage

def profile_memory(func):
    """
    Decorator to profile memory usage of a function
    
    Args:
        func: Function to profile
    
    Returns:
        Function wrapper that logs memory usage before and after execution
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Force garbage collection to get accurate memory usage
        gc.collect()
        
        # Log memory usage before function call
        mem_before = get_memory_usage()
        logging.info(f"Memory before {func.__name__}: {mem_before:.2f} MB")
        
        # Call the function
        result = func(*args, **kwargs)
        
        # Force garbage collection again
        gc.collect()
        
        # Log memory usage after function call
        mem_after = get_memory_usage()
        logging.info(f"Memory after {func.__name__}: {mem_after:.2f} MB (Delta: {mem_after - mem_before:.2f} MB)")
        
        return result
    
    return wrapper

# DataFrame optimization function

def optimize_dataframe(df, categorical_threshold=10, datetime_cols=None, verbose=False):
    """
    Optimize memory usage of a DataFrame by converting to appropriate data types
    
    Args:
        df (pd.DataFrame): DataFrame to optimize
        categorical_threshold (int): Maximum number of unique values to convert to categorical
        datetime_cols (list): List of datetime columns to convert
        verbose (bool): Whether to print detailed information
    
    Returns:
        pd.DataFrame: Optimized DataFrame
    """
    # Make a copy to avoid modifying the original
    result = df.copy()
    
    # Get initial memory usage
    initial_memory = result.memory_usage(deep=True).sum() / (1024 * 1024)  # MB
    
    if verbose:
        logging.info(f"Initial memory usage: {initial_memory:.2f} MB")
    
    # Process each column
    for col in result.columns:
        col_type = result[col].dtype
        
        # Numeric columns: Downcast to the smallest type that can represent the data
        if pd.api.types.is_integer_dtype(col_type):
            # Get min and max to determine the smallest possible type
            col_min = result[col].min()
            col_max = result[col].max()
            
            # Only proceed if we have actual data (not all NaN)
            if pd.notnull(col_min) and pd.notnull(col_max):
                if col_min >= 0:  # Unsigned int
                    if col_max <= 255:
                        result[col] = result[col].astype(np.uint8)
                    elif col_max <= 65535:
                        result[col] = result[col].astype(np.uint16)
                    elif col_max <= 4294967295:
                        result[col] = result[col].astype(np.uint32)
                else:  # Signed int
                    if col_min >= -128 and col_max <= 127:
                        result[col] = result[col].astype(np.int8)
                    elif col_min >= -32768 and col_max <= 32767:
                        result[col] = result[col].astype(np.int16)
                    elif col_min >= -2147483648 and col_max <= 2147483647:
                        result[col] = result[col].astype(np.int32)
        
        # Float columns: Downcast to float32 if possible
        elif pd.api.types.is_float_dtype(col_type):
            result[col] = result[col].astype(np.float32)
        
        # Object columns: Convert to categorical if few unique values
        elif pd.api.types.is_object_dtype(col_type) or pd.api.types.is_string_dtype(col_type):
            num_unique = result[col].nunique()
            total_values = len(result[col])
            
            # Convert to categorical if there are few unique values
            if num_unique > 0 and num_unique < min(categorical_threshold, total_values // 2):
                result[col] = result[col].astype('category')
    
    # Convert specified columns to datetime
    if datetime_cols:
        for col in datetime_cols:
            if col in result.columns:
                result[col] = pd.to_datetime(result[col], errors='coerce')
    
    # Get final memory usage
    final_memory = result.memory_usage(deep=True).sum() / (1024 * 1024)  # MB
    savings = 100 * (1 - final_memory / initial_memory)
    
    if verbose:
        logging.info(f"Final memory usage: {final_memory:.2f} MB")
        logging.info(f"Memory reduced by {savings:.1f}%")
    
    return result

# Progress bar and batch processing functions

class ProgressLogger:
    """
    Class to handle progress tracking with both tqdm and logging
    Can be used as a context manager or standalone
    """
    def __init__(self, total: int, desc: str = "", unit: str = "it", 
                 use_tqdm: bool = True, log_interval: int = 0):
        """
        Initialize the progress logger
        
        Args:
            total: Total number of items to process
            desc: Description text for the progress bar
            unit: Unit name for the items being processed
            use_tqdm: Whether to use tqdm for visual progress bar
            log_interval: Interval (in seconds) for logging progress (0 = no logging)
        """
        self.total = total
        self.desc = desc
        self.unit = unit
        self.use_tqdm = use_tqdm and TQDM_AVAILABLE
        self.log_interval = log_interval
        self.last_log_time = time.time()
        self.n = 0
        self.pbar = None
        
        if self.use_tqdm:
            self.pbar = tqdm(total=total, desc=desc, unit=unit)
    
    def update(self, n: int = 1):
        """Update progress by n units"""
        self.n += n
        
        if self.use_tqdm and self.pbar:
            self.pbar.update(n)
        
        # Log progress at specified intervals
        if self.log_interval > 0:
            current_time = time.time()
            if current_time - self.last_log_time >= self.log_interval:
                percentage = min(100.0, 100.0 * self.n / self.total if self.total > 0 else 100.0)
                logging.info(f"{self.desc}: {self.n}/{self.total} {self.unit} ({percentage:.1f}%)")
                self.last_log_time = current_time
    
    def set_description(self, desc: str):
        """Set a new description for the progress bar"""
        self.desc = desc
        if self.use_tqdm and self.pbar:
            self.pbar.set_description(desc)
    
    def close(self):
        """Close the progress bar"""
        if self.use_tqdm and self.pbar:
            self.pbar.close()
    
    def __enter__(self):
        """Enable use as a context manager"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close progress bar on context exit"""
        self.close()

def progress_map(func: Callable, items: List[Any], desc: str = "", unit: str = "it", 
                use_tqdm: bool = True, show_errors: bool = True, **kwargs) -> List[Any]:
    """
    Apply a function to each item with progress tracking
    
    Args:
        func: Function to apply to each item
        items: List of items to process
        desc: Description for the progress bar
        unit: Unit name for the items being processed
        use_tqdm: Whether to use tqdm progress bar
        show_errors: Whether to log errors (function will still continue on other items)
        **kwargs: Additional keyword arguments to pass to func
        
    Returns:
        List of results (None for items that raised exceptions)
    """
    results = []
    with ProgressLogger(total=len(items), desc=desc, unit=unit, use_tqdm=use_tqdm) as progress:
        for item in items:
            try:
                result = func(item, **kwargs)
                results.append(result)
            except Exception as e:
                if show_errors:
                    logging.error(f"Error processing {item}: {str(e)}")
                results.append(None)
            progress.update(1)
    
    return results

def batch_process(func: Callable, data_list: List[Any], batch_size: int = 10, 
                 desc: str = "Processing batches", use_tqdm: bool = True, 
                 *args, **kwargs) -> List[Any]:
    """
    Process a function in batches to control memory usage, with progress tracking
    
    Args:
        func: Function to call on each batch
        data_list: List of data items to process
        batch_size: Number of items to process in each batch
        desc: Description for the progress bar
        use_tqdm: Whether to use tqdm progress bar
        *args, **kwargs: Additional arguments to pass to func
    
    Returns:
        list: Combined results from all batches
    """
    results = []
    n_batches = (len(data_list) + batch_size - 1) // batch_size  # Ceiling division
    
    with ProgressLogger(total=n_batches, desc=desc, unit="batch", use_tqdm=use_tqdm) as progress:
        for i in range(n_batches):
            batch_start = i * batch_size
            batch_end = min(batch_start + batch_size, len(data_list))
            batch = data_list[batch_start:batch_end]
            
            # Process this batch
            progress.set_description(f"{desc} ({i+1}/{n_batches})")
            batch_result = func(batch, *args, **kwargs)
            
            # Store the result
            results.append(batch_result)
            
            # Force garbage collection to free memory
            gc.collect()
            
            progress.update(1)
    
    return results

# Pipeline progress tracking

class PipelineTracker:
    """
    Class to track progress of a multi-stage pipeline
    """
    def __init__(self, stages: List[str], use_tqdm: bool = True):
        """
        Initialize the pipeline tracker
        
        Args:
            stages: List of stage names in the pipeline
            use_tqdm: Whether to use tqdm for visual progress bar
        """
        self.stages = stages
        self.total_stages = len(stages)
        self.current_stage = 0
        self.use_tqdm = use_tqdm and TQDM_AVAILABLE
        self.overall_progress = None
        self.stage_progress = None
        
        # Initialize overall progress bar
        if self.use_tqdm:
            self.overall_progress = tqdm(total=self.total_stages, desc="Overall pipeline progress", 
                                        unit="stage", position=0, leave=True)
    
    def start_stage(self, stage_name: str, total_items: int = 0, unit: str = "it"):
        """
        Start a new pipeline stage
        
        Args:
            stage_name: Name of the stage (should match one in the stages list)
            total_items: Number of items to process in this stage (0 for unknown)
            unit: Unit name for the items being processed
        """
        # Close previous stage progress bar if exists
        if self.stage_progress:
            self.stage_progress.close()
            self.stage_progress = None
        
        # Find stage index
        try:
            stage_index = self.stages.index(stage_name)
            self.current_stage = stage_index
        except ValueError:
            # Stage not in predefined list, just use current stage
            pass
        
        # Log stage start
        logging.info(f"Starting pipeline stage {self.current_stage + 1}/{self.total_stages}: {stage_name}")
        
        # Create stage progress bar if items count provided
        if total_items > 0 and self.use_tqdm:
            self.stage_progress = tqdm(total=total_items, desc=f"Stage {self.current_stage + 1}: {stage_name}", 
                                      unit=unit, position=1, leave=True)
        
        return self.stage_progress
    
    def update_stage(self, n: int = 1):
        """Update current stage progress by n units"""
        if self.stage_progress:
            self.stage_progress.update(n)
    
    def complete_stage(self):
        """Mark current stage as complete"""
        # Close stage progress bar if exists
        if self.stage_progress:
            self.stage_progress.close()
            self.stage_progress = None
        
        # Update overall progress bar
        if self.overall_progress:
            self.overall_progress.update(1)
        
        # Log stage completion
        stage_name = self.stages[self.current_stage] if self.current_stage < len(self.stages) else "Unknown stage"
        logging.info(f"Completed pipeline stage {self.current_stage + 1}/{self.total_stages}: {stage_name}")
    
    def finish(self, success: bool = True):
        """
        Mark pipeline as finished
        
        Args:
            success: Whether the pipeline completed successfully
        """
        # Close all progress bars
        if self.stage_progress:
            self.stage_progress.close()
        
        if self.overall_progress:
            self.overall_progress.close()
        
        # Log completion status
        if success:
            logging.info(f"Pipeline completed successfully ({self.current_stage + 1}/{self.total_stages} stages)")
        else:
            logging.warning(f"Pipeline completed with errors (stopped at stage {self.current_stage + 1}/{self.total_stages})")
            
if __name__ == "__main__":
    # Example usage / testing code
    logging.info("Memory utilities module loaded")
    mem = memory_usage_report()
    logging.info(f"Current memory usage: {mem:.2f} MB")
    
    # Test progress tracking
    if TQDM_AVAILABLE:
        logging.info("Testing progress tracking functionality")
        
        # Test pipeline tracker
        pipeline = PipelineTracker(stages=["Data Processing", "Feature Engineering", "Model Training", "Evaluation"])
        
        # Stage 1
        stage1_progress = pipeline.start_stage("Data Processing", total_items=5)
        for i in range(5):
            # Simulate work
            time.sleep(0.5)
            pipeline.update_stage(1)
        pipeline.complete_stage()
        
        # Stage 2
        stage2_progress = pipeline.start_stage("Feature Engineering", total_items=3)
        for i in range(3):
            # Simulate work
            time.sleep(0.5)
            pipeline.update_stage(1)
        pipeline.complete_stage()
        
        # Finish pipeline
        pipeline.finish(success=True)
        
        # Test progress_map
        logging.info("Testing progress_map")
        def square(x):
            time.sleep(0.2)  # Simulate work
            return x * x
        
        results = progress_map(square, list(range(10)), desc="Squaring numbers", unit="number")
        logging.info(f"Results: {results}")
    
    # Done
    logging.info("Memory utilities test complete")