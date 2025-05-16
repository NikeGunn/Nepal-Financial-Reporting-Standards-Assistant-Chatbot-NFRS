"""
ASGI config for NFRS_Assistant project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/5.1/howto/deployment/asgi/
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Use the same settings module logic as in settings/__init__.py
environment = os.environ.get('DJANGO_SETTINGS_MODULE', 'NFRS_Assistant.settings.dev')
if environment in ['dev', 'prod']:
    os.environ['DJANGO_SETTINGS_MODULE'] = f'NFRS_Assistant.settings.{environment}'
else:
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", environment)

# Import Django first to initialize settings
from django.core.asgi import get_asgi_application
django_asgi_app = get_asgi_application()

# Now import Channels components after Django is initialized
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack

# Import websocket routing after Django setup to avoid import errors
try:
    from api.chat.routing import websocket_urlpatterns

    application = ProtocolTypeRouter({
        "http": django_asgi_app,
        "websocket": AuthMiddlewareStack(
            URLRouter(
                websocket_urlpatterns
            )
        ),
    })
except ImportError:
    # Fallback to HTTP-only if websocket routing is not available
    application = ProtocolTypeRouter({
        "http": django_asgi_app,
    })
