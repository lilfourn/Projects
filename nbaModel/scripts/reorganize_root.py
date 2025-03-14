#!/usr/bin/env python3
"""
Script to reorganize the root directory of the NBA model project.
This script will:
1. Create a 'scripts' directory for utility scripts
2. Create a 'docs' directory for documentation
3. Move files to their appropriate directories
4. Clean up temporary files
"""

import os
import sys
import shutil
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Define the project root
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Define directories to create
DIRECTORIES_TO_CREATE = [
    'scripts',  # For utility scripts
    'docs',     # For documentation
    'config',   # For configuration files
    'notebooks' # For Jupyter notebooks
]

# Define file mappings (source -> destination)
FILE_MAPPINGS = {
    # Scripts to scripts directory
    'cleanup_old_files.py': 'scripts/cleanup_old_files.py',
    'migrate_structure.py': 'scripts/migrate_structure.py',
    'test_new_structure.py': 'scripts/test_new_structure.py',
    
    # Documentation to docs directory
    'README_NEW_STRUCTURE.md': 'docs/README_NEW_STRUCTURE.md',
    'codebase-summary.md': 'docs/codebase-summary.md',
    
    # Keep these in the root
    'README.md': 'README.md',
    'requirements.txt': 'requirements.txt',
    'setup.py': 'setup.py',
    'main.py': 'main.py',
    
    # Config files to config directory
    '.repomixignore': 'config/.repomixignore',
    'repomix.config.json': 'config/repomix.config.json',
}

def create_directories():
    """Create the necessary directories if they don't exist."""
    for directory in DIRECTORIES_TO_CREATE:
        dir_path = os.path.join(PROJECT_ROOT, directory)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            logging.info(f"Created directory: {directory}")
        else:
            logging.info(f"Directory already exists: {directory}")

def move_files():
    """Move files to their appropriate directories."""
    moved_count = 0
    skipped_count = 0
    
    for source, destination in FILE_MAPPINGS.items():
        source_path = os.path.join(PROJECT_ROOT, source)
        dest_path = os.path.join(PROJECT_ROOT, destination)
        
        # Skip if source doesn't exist
        if not os.path.exists(source_path):
            logging.warning(f"Source file not found: {source}")
            skipped_count += 1
            continue
        
        # Skip if source and destination are the same
        if source == destination:
            logging.info(f"Keeping file in place: {source}")
            skipped_count += 1
            continue
        
        # Create destination directory if it doesn't exist
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        
        # Move the file
        try:
            shutil.move(source_path, dest_path)
            logging.info(f"Moved: {source} -> {destination}")
            moved_count += 1
        except Exception as e:
            logging.error(f"Error moving {source}: {str(e)}")
            skipped_count += 1
    
    return moved_count, skipped_count

def create_gitignore():
    """Create a .gitignore file if it doesn't exist."""
    gitignore_path = os.path.join(PROJECT_ROOT, '.gitignore')
    
    if not os.path.exists(gitignore_path):
        gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
ENV/

# IDE
.idea/
.vscode/
*.swp
*.swo

# Jupyter Notebooks
.ipynb_checkpoints

# Project specific
data/raw/
data/interim/
data/processed/
data/external/
models/*.pkl
models/*.joblib
logs/
"""
        
        with open(gitignore_path, 'w') as f:
            f.write(gitignore_content)
        
        logging.info("Created .gitignore file")

def create_readme_if_missing():
    """Create a README.md file if it doesn't exist."""
    readme_path = os.path.join(PROJECT_ROOT, 'README.md')
    
    if not os.path.exists(readme_path):
        readme_content = """# NBA Model

A machine learning model for predicting NBA player performance.

## Project Structure

```
nbaModel/
├── config/               # Configuration files
├── data/                 # Data files
│   ├── projections/      # Projection data
│   └── processed/        # Processed data
├── docs/                 # Documentation
├── models/               # Trained models
├── notebooks/            # Jupyter notebooks
├── scripts/              # Utility scripts
├── src/                  # Source code
│   ├── data_collection/  # Data collection scripts
│   ├── data_processing/  # Data processing scripts
│   ├── models/           # Model scripts
│   ├── utils/            # Utility scripts
│   └── visualization/    # Visualization scripts
├── main.py               # Main entry point
├── requirements.txt      # Project dependencies
└── setup.py              # Package setup script
```

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd nbaModel

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Fetch game data
python main.py fetch_game_data

# Get projections
python main.py get_projections

# Process data
python main.py process_data

# Train models
python main.py train_models

# Make predictions
python main.py predict

# Run the full pipeline
python main.py run_pipeline
```

## Documentation

See the [docs](docs/) directory for detailed documentation.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
"""
        
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        
        logging.info("Created README.md file")

def main():
    """Main function to execute the reorganization process."""
    logging.info("Starting reorganization of the root directory...")
    
    # Create directories
    create_directories()
    
    # Move files
    moved_count, skipped_count = move_files()
    
    # Create .gitignore file
    create_gitignore()
    
    # Create README.md if missing
    create_readme_if_missing()
    
    # Report results
    logging.info(f"\nReorganization completed:")
    logging.info(f"  - Created {len(DIRECTORIES_TO_CREATE)} directories")
    logging.info(f"  - Moved {moved_count} files")
    logging.info(f"  - Skipped {skipped_count} files")
    
    logging.info("\nThe root directory has been reorganized successfully!")

if __name__ == "__main__":
    main()
