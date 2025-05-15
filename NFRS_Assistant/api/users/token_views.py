"""
Custom token views with enhanced throttling for development environments.
"""
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView
from utils.throttling import AuthRateThrottle, TokenRefreshRateThrottle


class CustomTokenObtainPairView(TokenObtainPairView):
    """
    Custom token obtain view with dedicated throttling setting.
    This overrides the default TokenObtainPairView with our custom throttling.
    """
    throttle_classes = [AuthRateThrottle]


class CustomTokenRefreshView(TokenRefreshView):
    """
    Custom token refresh view with dedicated throttling setting.
    This overrides the default TokenRefreshView with our custom throttling.
    """
    throttle_classes = [TokenRefreshRateThrottle]