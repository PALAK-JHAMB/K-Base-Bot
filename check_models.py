import google.generativeai as genai
import yaml

# Load API key from your local config file
try:
    with open("config/settings.yaml", 'r') as f:
        config = yaml.safe_load(f)
    # Make sure your actual API key is in settings.yaml for this test
    api_key = config.get('gemini', {}).get('api_key')
    genai.configure(api_key=api_key)
except Exception as e:
    print(f"Error loading config or API key: {e}")
    exit()

print("--- Available Generative Models ---")
for m in genai.list_models():
  if 'generateContent' in m.supported_generation_methods:
    print(f"Model Name: {m.name}")