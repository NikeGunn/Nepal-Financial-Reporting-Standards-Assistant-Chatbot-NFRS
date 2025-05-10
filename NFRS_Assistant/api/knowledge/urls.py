from django.urls import path
from . import views

urlpatterns = [
    # Document management
    path('documents/', views.DocumentListCreateView.as_view(), name='document_list'),
    path('documents/<int:pk>/', views.DocumentDetailView.as_view(), name='document_detail'),

    # Document upload endpoints
    path('documents/upload/', views.DocumentUploadView.as_view(), name='document_upload'),
    path('documents/admin-upload/', views.AdminDocumentUploadView.as_view(), name='admin_document_upload'),

    # Vector search
    path('search/', views.VectorSearchView.as_view(), name='vector_search'),

    # Vector index management (admin only)
    path('indices/', views.VectorIndexListView.as_view(), name='vector_index_list'),
    path('indices/<int:pk>/', views.VectorIndexDetailView.as_view(), name='vector_index_detail'),
    path('indices/rebuild/', views.RebuildVectorIndexView.as_view(), name='rebuild_vector_index'),
]