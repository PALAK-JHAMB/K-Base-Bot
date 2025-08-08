# build.py
import sys
import os
import yaml

# --- System Path Setup ---
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.append(PROJECT_ROOT)

# --- Import the correct function from the correct file ---
from src.vector_store.vector_builder import build_vector_store

print("--- Starting the one-time vector store build process ---")

# Load the configuration from the local YAML file
settings_path = os.path.join(PROJECT_ROOT, "config", "settings.yaml")

try:
    with open(settings_path, 'r') as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    print(f"ERROR: Could not find {settings_path}. This build script requires a local config file.")
    sys.exit(1)

if not config.get('gemini', {}).get('api_key'):
     print(f"ERROR: 'api_key' not found in {settings_path}. The build script requires it.")
     sys.exit(1)

# --- Execute the build ---
build_vector_store(config)

print("--- Vector store build process complete. You can now switch back to app.py ---")