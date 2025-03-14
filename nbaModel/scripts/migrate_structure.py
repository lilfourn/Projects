#!/usr/bin/env python3
"""
Script to migrate the NBA model project to a new directory structure.
This script creates wrapper files in the root directory that import from the new structure,
ensuring backward compatibility while organizing the codebase.
"""

import os
import sys
import shutil
import re
from pathlib import Path

# Define the project root
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Define the mapping of files to their new locations
FILE_MAPPING = {
    # Data Collection
    'fetchGameData.py': 'src/data_collection/fetchGameData.py',
    'scrapeData.py': 'src/data_collection/scrapeData.py',
    'src/getProjections.py': 'src/data_collection/getProjections.py',
    
    # Data Processing
    'src/data_processing.py': 'src/data_processing/data_processing.py',
    'src/data_cleanup.py': 'src/data_processing/data_cleanup.py',
    'src/data_quality.py': 'src/data_processing/data_quality.py',
    'src/data_quality_check_derived.py': 'src/data_processing/data_quality_check_derived.py',
    'src/feature_engineering.py': 'src/data_processing/feature_engineering.py',
    'src/enhanced_defensive_features.py': 'src/data_processing/enhanced_defensive_features.py',
    
    # Models
    'src/model_builder.py': 'src/models/model_builder.py',
    'src/train_target_models.py': 'src/models/train_target_models.py',
    'src/predict.py': 'src/models/predict.py',
    'src/ensemble_models.py': 'src/models/ensemble_models.py',
    'train_all_targets.py': 'src/models/train_all_targets.py',
    'train_xgboost_models.py': 'src/models/train_xgboost_models.py',
    'predict.py': 'src/models/predict.py',
    
    # Visualization
    'src/feature_viz.py': 'src/visualization/feature_viz.py',
    'src/feature_importance_viz.py': 'src/visualization/feature_importance_viz.py',
    'src/model_comparison.py': 'src/visualization/model_comparison.py',
    'analyze_projections.py': 'src/visualization/analyze_projections.py',
    
    # Utils
    'src/config.py': 'src/utils/config.py',
    'src/memory_utils.py': 'src/utils/memory_utils.py',
    'src/run_pipeline.py': 'src/utils/run_pipeline.py',
}

# Template for wrapper files
WRAPPER_TEMPLATE = '''#!/usr/bin/env python3
"""
Wrapper script to maintain backward compatibility.
This script imports and runs the {module_name} module from the new directory structure.
"""

import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the main function from the module
from {new_import_path} import main

if __name__ == "__main__":
    main()
'''

def create_directory_structure():
    """Create the new directory structure."""
    directories = [
        'src/data_collection',
        'src/data_processing',
        'src/models',
        'src/visualization',
        'src/utils',
    ]
    
    for directory in directories:
        os.makedirs(os.path.join(PROJECT_ROOT, directory), exist_ok=True)
        # Create __init__.py file in each directory
        init_path = os.path.join(PROJECT_ROOT, directory, '__init__.py')
        if not os.path.exists(init_path):
            with open(init_path, 'w') as f:
                f.write(f'"""\n{directory.split("/")[-1].replace("_", " ").title()} module for NBA model.\n"""')

def copy_files():
    """Copy files to their new locations."""
    for source, destination in FILE_MAPPING.items():
        source_path = os.path.join(PROJECT_ROOT, source)
        dest_path = os.path.join(PROJECT_ROOT, destination)
        
        # Skip if source doesn't exist
        if not os.path.exists(source_path):
            print(f"Warning: Source file {source_path} does not exist. Skipping.")
            continue
        
        # Create destination directory if it doesn't exist
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        
        # Copy the file
        shutil.copy2(source_path, dest_path)
        print(f"Copied {source} to {destination}")

def create_wrapper_files():
    """Create wrapper files in the root directory for backward compatibility."""
    root_files = [f for f in FILE_MAPPING.keys() if '/' not in f]
    
    for file in root_files:
        new_path = FILE_MAPPING[file]
        module_name = os.path.splitext(file)[0]
        new_import_path = new_path.replace('/', '.').replace('.py', '')
        
        wrapper_content = WRAPPER_TEMPLATE.format(
            module_name=module_name,
            new_import_path=new_import_path
        )
        
        wrapper_path = os.path.join(PROJECT_ROOT, f"{module_name}_wrapper.py")
        with open(wrapper_path, 'w') as f:
            f.write(wrapper_content)
        
        print(f"Created wrapper file {wrapper_path}")

def update_imports():
    """Update import statements in the copied files to reflect the new structure."""
    for _, destination in FILE_MAPPING.items():
        file_path = os.path.join(PROJECT_ROOT, destination)
        
        if not os.path.exists(file_path):
            continue
        
        try:
            # Try different encodings to handle potential encoding issues
            encodings = ['utf-8', 'latin-1', 'cp1252']
            content = None
            
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read()
                    break  # If successful, break out of the loop
                except UnicodeDecodeError:
                    continue
            
            if content is None:
                print(f"Warning: Could not read {destination} with any of the attempted encodings. Skipping.")
                continue
            
            # Update imports from src.X to src.category.X
            # This is a simplified approach and may need manual adjustments
            for old_path, new_path in FILE_MAPPING.items():
                if old_path.startswith('src/'):
                    old_module = old_path.replace('src/', '').replace('.py', '')
                    new_module = new_path.replace('src/', '').replace('.py', '')
                    
                    # Replace import statements
                    content = re.sub(
                        r'from\s+src\.{0}\s+import'.format(old_module),
                        'from src.{0} import'.format(new_module),
                        content
                    )
                    content = re.sub(
                        r'import\s+src\.{0}'.format(old_module),
                        'import src.{0}'.format(new_module),
                        content
                    )
            
            # Write the updated content back to the file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"Updated imports in {destination}")
        except Exception as e:
            print(f"Error updating imports in {destination}: {str(e)}")

def main():
    """Main function to execute the migration process."""
    print("Starting NBA model project structure migration...")
    
    # Create the new directory structure
    create_directory_structure()
    
    # Copy files to their new locations
    copy_files()
    
    # Create wrapper files for backward compatibility
    create_wrapper_files()
    
    # Update import statements
    update_imports()
    
    print("\nMigration completed successfully!")
    print("\nNotes:")
    print("1. The original files have been preserved.")
    print("2. Wrapper files have been created with '_wrapper' suffix.")
    print("3. You may need to manually adjust some import statements.")
    print("4. Once everything is working, you can remove the original files.")

if __name__ == "__main__":
    main()
