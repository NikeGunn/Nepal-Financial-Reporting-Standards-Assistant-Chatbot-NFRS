"""
Custom throttling classes for rate limiting.
"""
from rest_framework.throttling import ScopedRateThrottle


class AuthRateThrottle(ScopedRateThrottle):
    """
    Throttle class specifically for authentication-related views.
    This applies more lenient limits than general API endpoints.
    """
    scope = 'auth'


class TokenRefreshRateThrottle(ScopedRateThrottle):
    """
    Throttle class specifically for token refresh operations.
    """
    scope = 'token_refresh'