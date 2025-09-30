# build.py
import sys
import os
import yaml

# Add the project's root directory to the Python path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Now we can import our builder function
from src.vector_store.vector_builder import build_vector_store

print("--- Starting the one-time vector store build process ---")

# Load the configuration from the local YAML file
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
settings_path = os.path.join(PROJECT_ROOT, "config", "settings.yaml")

# This script requires a local settings.yaml
try:
    with open(settings_path, 'r') as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    print(f"ERROR: Could not find {settings_path}. This build script requires a local config file.")
    sys.exit(1)

# Execute the build
build_vector_store(config)

print("--- Vector store build process complete. You can now switch your app file back to src/ui/app.py ---")