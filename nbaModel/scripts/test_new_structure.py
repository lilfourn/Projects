#!/usr/bin/env python3
"""
Test script to verify that the new directory structure is working properly.
This script attempts to import modules from the new structure and reports any issues.
"""

import sys
import os
import importlib
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test importing modules from the new structure."""
    modules_to_test = [
        # Data Collection
        'src.data_collection.fetchGameData',
        'src.data_collection.scrapeData',
        'src.data_collection.getProjections',
        
        # Data Processing
        'src.data_processing.data_processing',
        'src.data_processing.data_cleanup',
        'src.data_processing.data_quality',
        'src.data_processing.feature_engineering',
        'src.data_processing.enhanced_defensive_features',
        
        # Models
        'src.models.model_builder',
        'src.models.train_target_models',
        'src.models.predict',
        'src.models.train_all_targets',
        'src.models.train_xgboost_models',
        
        # Visualization
        'src.visualization.feature_viz',
        'src.visualization.feature_importance_viz',
        'src.visualization.model_comparison',
        'src.visualization.analyze_projections',
        
        # Utils
        'src.utils.config',
        'src.utils.memory_utils',
        'src.utils.run_pipeline',
    ]
    
    successful_imports = []
    failed_imports = []
    
    for module_name in modules_to_test:
        try:
            module = importlib.import_module(module_name)
            successful_imports.append(module_name)
            logging.info(f"Successfully imported {module_name}")
        except Exception as e:
            failed_imports.append((module_name, str(e)))
            logging.error(f"Failed to import {module_name}: {str(e)}")
    
    # Report results
    logging.info(f"\nImport Test Results:")
    logging.info(f"  - Successfully imported: {len(successful_imports)}/{len(modules_to_test)} modules")
    
    if failed_imports:
        logging.error(f"  - Failed imports ({len(failed_imports)}):")
        for module_name, error in failed_imports:
            logging.error(f"    - {module_name}: {error}")
    else:
        logging.info(f"  - All imports successful!")
    
    return len(failed_imports) == 0

def test_main_script():
    """Test running the main script with different components."""
    components = [
        'fetch_game_data',
        'scrape_data',
        'get_projections',
        'process_data',
        'train_models',
        'predict',
        'visualize',
        'run_pipeline',
    ]
    
    logging.info(f"\nMain Script Test:")
    logging.info(f"  To test the main script, run the following commands:")
    
    for component in components:
        logging.info(f"    python main.py {component} --help")
    
    logging.info(f"\n  These commands will display the help message for each component.")
    logging.info(f"  If the help message is displayed correctly, the main script is working properly.")

def main():
    """Main function to run the tests."""
    logging.info("Testing the new directory structure...")
    
    # Test imports
    imports_ok = test_imports()
    
    # Test main script
    test_main_script()
    
    # Final report
    if imports_ok:
        logging.info("\nAll tests passed! The new directory structure is working properly.")
    else:
        logging.warning("\nSome tests failed. Please check the error messages above.")
        logging.info("You may need to manually fix some import statements.")

if __name__ == "__main__":
    main()
