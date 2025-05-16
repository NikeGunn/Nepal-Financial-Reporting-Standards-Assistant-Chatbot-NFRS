from django.urls import path
from .views import (
    DocumentListCreateView, DocumentDetailView, DocumentUploadView,
    AdminDocumentUploadView, VectorSearchView, VectorIndexListView,
    VectorIndexDetailView, RebuildVectorIndexView,
    # New session-based document views
    SessionDocumentListCreateView, SessionDocumentDetailView,
    SessionDocumentSearchView, CleanupSessionDocumentsView
)

urlpatterns = [
    # Standard document endpoints
    path('documents/', DocumentListCreateView.as_view(), name='document-list'),
    path('documents/<int:pk>/', DocumentDetailView.as_view(), name='document-detail'),
    path('documents/upload/', DocumentUploadView.as_view(), name='document-upload'),
    path('documents/admin-upload/', AdminDocumentUploadView.as_view(), name='admin-document-upload'),

    # Vector search endpoint
    path('search/vector/', VectorSearchView.as_view(), name='vector-search'),

    # Vector index management endpoints
    path('indices/', VectorIndexListView.as_view(), name='vector-index-list'),
    path('indices/<int:pk>/', VectorIndexDetailView.as_view(), name='vector-index-detail'),
    path('indices/rebuild/', RebuildVectorIndexView.as_view(), name='rebuild-vector-index'),

    # Session-based document endpoints
    path('session-documents/', SessionDocumentListCreateView.as_view(), name='session-document-list'),
    path('session-documents/<int:pk>/', SessionDocumentDetailView.as_view(), name='session-document-detail'),
    path('session-documents/search/', SessionDocumentSearchView.as_view(), name='session-document-search'),
    path('session-documents/cleanup/', CleanupSessionDocumentsView.as_view(), name='cleanup-session-documents'),
]