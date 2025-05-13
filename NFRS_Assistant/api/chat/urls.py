from django.urls import path
from . import views

urlpatterns = [
    # Conversation management
    path('conversations/', views.ConversationListCreateView.as_view(), name='conversation_list'),
    path('conversations/<int:pk>/', views.ConversationDetailView.as_view(), name='conversation_detail'),
    path('user-conversations/', views.UserConversationsView.as_view(), name='user_conversations'),

    # Chat messaging
    path('messages/', views.ChatMessageView.as_view(), name='chat_message'),
    path('messages/<int:pk>/', views.MessageDetailView.as_view(), name='message_detail'),

    # Translation
    path('translate/', views.TranslateMessageView.as_view(), name='translate_message'),
]