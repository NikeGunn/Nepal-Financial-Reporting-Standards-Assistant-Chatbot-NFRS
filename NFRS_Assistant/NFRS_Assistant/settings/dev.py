"""
Development settings for NFRS_Assistant project.
"""

from .base import *

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

# Database
# Using SQLite for development for simplicity
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

# For development, you might want to add Django Debug Toolbar
INSTALLED_APPS += [
    # 'debug_toolbar',
]

MIDDLEWARE += [
    # 'debug_toolbar.middleware.DebugToolbarMiddleware',
]

# Django Debug Toolbar settings
INTERNAL_IPS = [
    '127.0.0.1',
]

# Email backend for development
EMAIL_BACKEND = 'django.core.mail.backends.console.EmailBackend'

# Override throttle rates for development with much more permissive values
REST_FRAMEWORK['DEFAULT_THROTTLE_RATES'] = {
    'anon': '1000/day',        # Increased from 100/day
    'user': '5000/day',        # Increased from 1000/day
    'auth': '1000/hour',
    'token_refresh': '500/hour' # Increased from 100/hour
}