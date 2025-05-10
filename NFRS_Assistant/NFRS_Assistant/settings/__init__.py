"""
Initialize settings based on environment variable.
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Default to 'dev' settings if DJANGO_SETTINGS_MODULE is not set
environment = os.environ.get('DJANGO_SETTINGS_MODULE', 'NFRS_Assistant.settings.dev')

# If only the environment name is provided, construct the full module path
if environment in ['dev', 'prod']:
    os.environ['DJANGO_SETTINGS_MODULE'] = f'NFRS_Assistant.settings.{environment}'