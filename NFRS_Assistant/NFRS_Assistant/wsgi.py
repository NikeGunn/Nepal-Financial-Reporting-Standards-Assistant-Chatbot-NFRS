"""
WSGI config for NFRS_Assistant project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/5.1/howto/deployment/wsgi/
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from django.core.wsgi import get_wsgi_application

# Use the same settings module logic as in settings/__init__.py
environment = os.environ.get('DJANGO_SETTINGS_MODULE', 'NFRS_Assistant.settings.dev')
if environment in ['dev', 'prod']:
    os.environ['DJANGO_SETTINGS_MODULE'] = f'NFRS_Assistant.settings.{environment}'
else:
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", environment)

application = get_wsgi_application()
