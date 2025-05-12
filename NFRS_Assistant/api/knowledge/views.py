from rest_framework import generics, status, permissions
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser
from django.conf import settings
from django.db import transaction
import os
import fitz  # PyMuPDF
import openai
import numpy as np
import json
import magic
import uuid
import pickle
import subprocess
import threading
import logging  # Added import for logging
from sklearn.metrics.pairwise import cosine_similarity
from .models import Document, DocumentChunk, VectorIndex
from .serializers import (
    DocumentSerializer, DocumentChunkSerializer, DocumentUploadSerializer,
    VectorIndexSerializer, SearchQuerySerializer
)
from utils.vector_ops import create_embedding, update_index_with_chunks, vector_search

# Setup logger
logger = logging.getLogger(__name__)

# Helper function to run document processing in background
def process_document_async(document_id):
    """Run document processing in a separate thread."""
    def run_processing():
        try:
            from utils.document_processor import process_document
            from .models import Document

            # Update document status before processing
            try:
                document = Document.objects.get(id=document_id)
                document.processing_status = 'processing'
                document.save()
            except Exception as e:
                print(f"Error updating document status before processing: {e}")

            # Process the document
            process_document(document_id)
        except Exception as e:
            print(f"Error in background processing thread: {e}")

            # Try to update document status on error
            try:
                from .models import Document
                document = Document.objects.get(id=document_id)
                document.processing_status = 'failed'
                document.error_message = str(e)[:500]  # Limit error message length
                document.save()
            except Exception as nested_error:
                print(f"Failed to update document status after error: {nested_error}")

    # Start background thread and detach it
    thread = threading.Thread(target=run_processing)
    thread.daemon = True  # Thread will terminate when main program exits
    thread.start()

    # Return thread handle for testing/debugging
    return thread


class IsAdminUser(permissions.BasePermission):
    """
    Custom permission to only allow admin users to access.
    """
    def has_permission(self, request, view):
        if not request.user or not request.user.is_authenticated:
            return False
        return hasattr(request.user, 'profile') and request.user.profile.is_admin


class DocumentListCreateView(generics.ListCreateAPIView):
    """
    API endpoint for listing and creating documents.
    """
    serializer_class = DocumentSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        user = self.request.user
        # Admin users can see all documents, regular users see only public docs and their own
        if hasattr(user, 'profile') and user.profile.is_admin:
            return Document.objects.all()
        return Document.objects.filter(
            uploaded_by=user
        ) | Document.objects.filter(
            is_public=True
        )


class DocumentDetailView(generics.RetrieveUpdateDestroyAPIView):
    """
    API endpoint for retrieving, updating, and deleting documents.
    """
    serializer_class = DocumentSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        # Check if this is a schema generation request
        if getattr(self, 'swagger_fake_view', False):
            return Document.objects.none()  # Return empty queryset for schema generation

        user = self.request.user
        if hasattr(user, 'profile') and user.profile.is_admin:
            return Document.objects.all()
        return Document.objects.filter(
            uploaded_by=user
        ) | Document.objects.filter(
            is_public=True
        )


class DocumentUploadView(APIView):
    """
    API endpoint for uploading documents.
    """
    parser_classes = [MultiPartParser, FormParser]
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request):
        try:
            # Set a moderate timeout for regular uploads
            request.upload_handlers[0].chunk_size = 5 * 1024 * 1024  # 5MB chunk size

            # Process the upload request
            serializer = DocumentUploadSerializer(data=request.data, context={'request': request})
            if serializer.is_valid():
                # Create and save document with pending status
                document = serializer.save(processing_status='pending')

                # Import the processor utility directly
                from utils.document_processor import process_document_async

                # Use a unique flag to prevent duplicate processing attempts
                processing_flag = f"doc_{document.id}_processing_{uuid.uuid4().hex[:8]}"

                # Log the processing attempt
                logger.info(f"Scheduling document {document.id} '{document.title}' for processing with flag: {processing_flag}")

                # Schedule background processing - only when transaction completes
                transaction.on_commit(lambda: process_document_async(document.id))

                # Return success immediately to avoid client timeout
                response_data = DocumentSerializer(document).data
                response_data['status'] = 'success'
                response_data['message'] = 'Document uploaded successfully and is being processed in the background.'

                return Response(
                    response_data,
                    status=status.HTTP_201_CREATED
                )

            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        except Exception as e:
            # Log error without trying to import logging again
            logger.error(f"Error in DocumentUploadView: {str(e)}")

            # Catch any unexpected errors during upload and return a friendly message
            return Response(
                {"error": f"Document upload failed: {str(e)}", "status": "error"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class AdminDocumentUploadView(DocumentUploadView):
    """
    API endpoint for admin document uploads.
    Handles large document uploads with improved timeout handling.
    """
    permission_classes = [permissions.IsAuthenticated, IsAdminUser]

    def post(self, request):
        try:
            # Set a custom timeout for large document uploads
            request.upload_handlers[0].chunk_size = 10 * 1024 * 1024  # Increase chunk size to 10MB

            # Set a longer timeout for large uploads
            if hasattr(settings, 'DATA_UPLOAD_MAX_MEMORY_SIZE'):
                # Temporarily increase max memory size for this request
                original_max_size = settings.DATA_UPLOAD_MAX_MEMORY_SIZE
                settings.DATA_UPLOAD_MAX_MEMORY_SIZE = 100 * 1024 * 1024  # 100MB

            # Process the upload request
            serializer = DocumentUploadSerializer(data=request.data, context={'request': request})
            if serializer.is_valid():
                document = serializer.save()

                # Set document to pending status for background processing
                document.processing_status = 'pending'
                document.save()
                
                # Import the processor utility directly 
                from utils.document_processor import process_document_async
                
                # Use a unique identifier for tracking this processing job
                process_id = uuid.uuid4().hex[:8]
                logger.info(f"Admin document upload: Scheduling document {document.id} '{document.title}' for processing with ID: {process_id}")

                # Start background processing in a fully detached manner
                # Schedule the async processing to happen after the response is sent
                transaction.on_commit(lambda: process_document_async(document.id))

                # Create the response with status info specifically for API clients
                response_data = DocumentSerializer(document).data
                response_data['status'] = 'success'
                response_data['message'] = 'Document uploaded successfully and is being processed in the background.'

                # Return a standard response without problematic hop-by-hop headers
                return Response(
                    response_data,
                    status=status.HTTP_201_CREATED
                )

            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

            # Reset the setting if we changed it
            if hasattr(settings, 'DATA_UPLOAD_MAX_MEMORY_SIZE') and 'original_max_size' in locals():
                settings.DATA_UPLOAD_MAX_MEMORY_SIZE = original_max_size

        except Exception as e:
            # Log the error for debugging using the module-level logger
            logger.error(f"Error in AdminDocumentUploadView: {str(e)}")

            # Return a more informative error response
            return Response(
                {"error": f"Admin document upload failed: {str(e)}", "status": "error"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class VectorSearchView(APIView):
    """
    API endpoint for searching documents using vector similarity.
    """
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request):
        serializer = SearchQuerySerializer(data=request.data)
        if serializer.is_valid():
            query = serializer.validated_data['query']
            top_k = serializer.validated_data.get('top_k', 5)
            filter_document_ids = serializer.validated_data.get('filter_document_ids', [])

            try:
                # Use the vector_search utility function
                results = vector_search(query, top_k=top_k * 2, filter_document_ids=filter_document_ids)

                if not results:
                    return Response(
                        {"error": "No search results found or vector index issue"},
                        status=status.HTTP_404_NOT_FOUND
                    )

                # Get complete document metadata for the results
                document_ids = list(set(r['document_id'] for r in results))
                documents = Document.objects.filter(id__in=document_ids)
                document_data = {
                    doc.id: {
                        'title': doc.title,
                        'description': doc.description,
                        'file_type': doc.file_type
                    } for doc in documents
                }

                # Add document info to results
                for result in results:
                    doc_id = result['document_id']
                    if doc_id in document_data:
                        result['document'] = document_data[doc_id]

                # Limit to requested top_k
                results = results[:top_k]

                return Response({
                    'query': query,
                    'results': results
                })

            except Exception as e:
                return Response(
                    {"error": str(e)},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class VectorIndexListView(generics.ListAPIView):
    """
    API endpoint for listing vector indices (admin only).
    """
    serializer_class = VectorIndexSerializer
    permission_classes = [permissions.IsAuthenticated, IsAdminUser]
    queryset = VectorIndex.objects.all()


class VectorIndexDetailView(generics.RetrieveUpdateAPIView):
    """
    API endpoint for retrieving and updating vector indices (admin only).
    """
    serializer_class = VectorIndexSerializer
    permission_classes = [permissions.IsAuthenticated, IsAdminUser]
    queryset = VectorIndex.objects.all()


class RebuildVectorIndexView(APIView):
    """
    API endpoint for rebuilding the vector index (admin only).
    """
    permission_classes = [permissions.IsAuthenticated, IsAdminUser]

    def post(self, request):
        try:
            # Create a new index
            index_name = f"nfrs_index_{uuid.uuid4().hex[:8]}"
            index_file_path = os.path.join(settings.VECTOR_STORE_DIR, f"{index_name}.index")
            metadata_path = os.path.join(settings.VECTOR_STORE_DIR, f"{index_name}_metadata.json")

            # Create directory if it doesn't exist
            os.makedirs(settings.VECTOR_STORE_DIR, exist_ok=True)

            # Get all chunks with embeddings
            chunks = DocumentChunk.objects.filter(embedding_vector__isnull=False)

            # Extract vectors and metadata
            vectors = []
            metadata = []

            for chunk in chunks:
                vector = np.frombuffer(chunk.embedding_vector, dtype=np.float32)
                vectors.append(vector)

                metadata.append({
                    'chunk_id': chunk.id,
                    'document_id': chunk.document_id,
                    'content': chunk.content[:500],  # Store a preview of the content
                    'page_number': chunk.page_number
                })

            # Create scikit-learn compatible array
            if vectors:
                vectors_array = np.vstack(vectors)

                # Save index using pickle (scikit-learn compatible)
                with open(index_file_path, 'wb') as f:
                    pickle.dump(vectors_array, f)

                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f)

            # Deactivate old indices
            VectorIndex.objects.all().update(is_active=False)

            # Create new index record
            vector_index = VectorIndex.objects.create(
                name=index_name,
                description="Rebuilt NFRS documents vector index",
                index_file_path=index_file_path,
                num_vectors=len(metadata),
                is_active=True
            )

            return Response({
                "message": "Vector index rebuilt successfully",
                "index_id": vector_index.id,
                "vectors_count": len(metadata)
            })

        except Exception as e:
            return Response(
                {"error": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )