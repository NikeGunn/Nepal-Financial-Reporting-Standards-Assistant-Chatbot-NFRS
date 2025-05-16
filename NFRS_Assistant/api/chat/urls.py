from django.urls import path
from rest_framework.routers import DefaultRouter
from . import views

router = DefaultRouter()
router.register(r'conversations', views.ConversationViewSet, basename='conversation')

urlpatterns = [
    # Chat messages
    path('messages/', views.ChatMessageView.as_view({'post': 'create'}), name='chat_message'),
    path('messages/translate/', views.ChatMessageView.as_view({'post': 'translate'}), name='translate_message'),
    path('messages/session-documents/', views.ChatMessageView.as_view({'get': 'session_documents'}), name='session_documents'),

    # Add compatibility URL for ConversationListCreateView
    path('conversations-list/', views.ConversationListCreateView.as_view(), name='conversation-list-create'),

    # User conversations endpoint
    path('user-conversations/', views.ConversationViewSet.as_view({'get': 'list'}), name='user-conversations'),
]

urlpatterns += router.urls