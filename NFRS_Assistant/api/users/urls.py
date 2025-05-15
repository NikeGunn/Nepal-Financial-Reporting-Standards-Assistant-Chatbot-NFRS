from django.urls import path
from . import views
from .token_views import CustomTokenObtainPairView, CustomTokenRefreshView

urlpatterns = [
    # Authentication endpoints
    path('token/', CustomTokenObtainPairView.as_view(), name='token_obtain_pair'),
    path('token/refresh/', CustomTokenRefreshView.as_view(), name='token_refresh'),
    path('register/', views.UserRegistrationView.as_view(), name='register'),

    # User profile endpoints
    path('profile/', views.UserProfileView.as_view(), name='profile'),
    path('change-password/', views.ChangePasswordView.as_view(), name='change_password'),

    # API key management
    path('api-keys/', views.ApiKeyListCreateView.as_view(), name='api_key_list'),
    path('api-keys/<int:pk>/', views.ApiKeyDetailView.as_view(), name='api_key_detail'),
]