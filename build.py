# build.py
import sys
import os
import yaml

# Add the project's root directory to the Python path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Now we can import our builder function
from src.vector_store.vector_builder import create_knowledge_base # Using the renamed function

print("--- Starting the knowledge base build process ---")

# Load the configuration from the local YAML file
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
settings_path = os.path.join(PROJECT_ROOT, "config", "settings.yaml")

# This script requires a local settings.yaml and a real API key in it
# because we cannot access st.secrets here.
try:
    with open(settings_path, 'r') as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    print(f"ERROR: Could not find {settings_path}. This build script requires a local config file.")
    sys.exit(1)

if not config.get('gemini', {}).get('api_key'):
     print(f"ERROR: 'api_key' not found in {settings_path}. The build script requires it.")
     sys.exit(1)

create_knowledge_base(config)
print("--- Knowledge base build process complete. ---")