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
import tempfile
from sklearn.metrics.pairwise import cosine_similarity
from .models import Document, DocumentChunk, VectorIndex, SessionDocument, SessionDocumentChunk
from .serializers import (
    DocumentSerializer, DocumentChunkSerializer, DocumentUploadSerializer,
    VectorIndexSerializer, SearchQuerySerializer, SessionDocumentSerializer,
    SessionDocumentUploadSerializer
)
from utils.vector_ops import create_embedding, update_index_with_chunks, vector_search
from utils.document_processor import (
    process_document_async, process_session_document,
    vector_search_session_documents, cleanup_session_documents
)

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

def process_session_document_async(file_path, session_id, chat_id=None, title=None, user=None, document_id=None):
    """Run session document processing in a separate thread."""
    def run_processing():
        # Create a separate database connection for this thread to avoid connection sharing issues
        from django.db import connection, DatabaseError
        connection.close()  # Close the connection shared from the main thread

        try:
            from utils.document_processor import extract_text_from_file, create_text_chunks, create_embedding
            from .models import SessionDocument, SessionDocumentChunk
            from django.db.models import ObjectDoesNotExist

            # If we have a document_id, we need to update the existing document
            if document_id:
                try:
                    # First check if document still exists before doing any processing
                    try:
                        # Use a separate transaction to verify document existence
                        session_doc = SessionDocument.objects.get(id=document_id)
                    except (ObjectDoesNotExist, DatabaseError):
                        # Document has been deleted, stop processing
                        logger.info(f"Skipping processing for deleted document ID: {document_id}")
                        return

                    # Extract text from the document
                    extracted_texts = extract_text_from_file(file_path)
                    if not extracted_texts:
                        logger.error(f"Failed to extract text from session document: {file_path}")
                        return

                    # Create a content preview
                    content_preview = ""
                    for text_data in extracted_texts[:2]:  # Just use first couple of pages for preview
                        content_preview += text_data['text'][:250]
                        if len(content_preview) >= 250:
                            content_preview = content_preview[:250] + "..."
                            break

                    # Verify document still exists before updating
                    try:
                        session_doc = SessionDocument.objects.get(id=document_id)
                        session_doc.content_preview = content_preview
                        session_doc.save()
                    except (ObjectDoesNotExist, DatabaseError):
                        # Document has been deleted during processing
                        logger.info(f"Document {document_id} was deleted during processing - skipping update")
                        return

                    # Process each extracted text into chunks with embeddings
                    chunks_created = 0

                    for idx, extracted in enumerate(extracted_texts):
                        # Check if document still exists periodically
                        if idx % 5 == 0:  # Check every 5 chunks
                            try:
                                # Just check existence, don't retrieve full object
                                if not SessionDocument.objects.filter(id=document_id).exists():
                                    logger.info(f"Document {document_id} was deleted during processing - stopping chunk creation")
                                    return
                            except DatabaseError:
                                # Database error, likely due to deletion
                                return

                        text = extracted['text']
                        page = extracted.get('page', idx + 1)

                        if not text.strip():
                            continue

                        # Split text into chunks
                        chunks = create_text_chunks(text)

                        # Create chunks with embeddings
                        for chunk_idx, chunk_text in enumerate(chunks):
                            if not chunk_text.strip():
                                continue

                            try:
                                # Double-check the document still exists before each chunk creation
                                if not SessionDocument.objects.filter(id=document_id).exists():
                                    return
                                
                                # Generate embedding
                                embedding = create_embedding(chunk_text)

                                # Create the session document chunk
                                chunk = SessionDocumentChunk(
                                    session_document_id=document_id,  # Use direct ID assignment to avoid foreign key issues
                                    content=chunk_text,
                                    chunk_index=chunks_created,
                                    page_number=page
                                )

                                if embedding is not None:
                                    chunk.embedding_vector = embedding.tobytes()

                                chunk.save()
                                chunks_created += 1

                            except ObjectDoesNotExist:
                                # Document was deleted, stop processing silently
                                return
                            except DatabaseError as db_error:
                                # Likely foreign key error due to document being deleted
                                if "FOREIGN KEY constraint failed" in str(db_error):
                                    logger.debug(f"Document {document_id} no longer exists, stopping processing")
                                    return
                                logger.error(f"Database error processing chunk: {str(db_error)}")
                                return
                            except Exception as chunk_error:
                                # Don't log foreign key errors, as these are expected when documents are deleted
                                if "FOREIGN KEY constraint failed" not in str(chunk_error):
                                    logger.error(f"Error processing session document chunk: {str(chunk_error)}")
                                return  # Stop processing on any error

                    logger.info(f"Successfully processed session document: {document_id} with {chunks_created} chunks")
                except ObjectDoesNotExist:
                    # Document doesn't exist anymore, log at debug level only since this is expected during cleanup
                    logger.debug(f"Session document with ID {document_id} not found, was likely deleted")
                except DatabaseError as db_error:
                    # Likely foreign key error, which we expect if document was deleted
                    if "FOREIGN KEY constraint failed" in str(db_error):
                        logger.debug(f"Document {document_id} was deleted during processing")
                    else:
                        logger.error(f"Database error in document processing: {str(db_error)}")
                except Exception as e:
                    # Non-database errors should still be logged
                    logger.error(f"Error in document processing: {str(e)}")

            else:
                # Process as new document (fallback to original behavior)
                from utils.document_processor import process_session_document
                process_session_document(
                    file_path=file_path,
                    session_id=session_id,
                    chat_id=chat_id,
                    title=title,
                    user=user
                )

        except Exception as e:
            # Only log if not a foreign key constraint failure
            if "FOREIGN KEY constraint failed" not in str(e):
                logger.error(f"Error in session document processing thread: {e}")
        finally:
            # Always try to delete the temp file, even if processing fails
            try:
                if os.path.exists(file_path):
                    os.unlink(file_path)
            except Exception as e:
                logger.error(f"Error deleting temporary file {file_path}: {e}")

            # Explicitly close the database connection when done
            try:
                connection.close()
            except Exception:
                pass

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


class SessionDocumentListCreateView(APIView):
    """
    API endpoint for listing and uploading session-based documents.
    """
    parser_classes = [MultiPartParser, FormParser]
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        """List all session documents for a session or chat."""
        # Get query parameters
        session_id = request.query_params.get('session_id')
        chat_id = request.query_params.get('chat_id')

        # Require at least one filter parameter
        if not session_id and not chat_id:
            return Response(
                {"error": "Either session_id or chat_id query parameter is required"},
                status=status.HTTP_400_BAD_REQUEST
            )

        # Build filters
        filters = {}
        if session_id:
            filters['session_id'] = session_id
        if chat_id:
            filters['chat_id'] = chat_id

        # Get documents
        documents = SessionDocument.objects.filter(**filters)
        serializer = SessionDocumentSerializer(documents, many=True)

        return Response(serializer.data)

    def post(self, request):
        """Upload a new session document."""
        try:
            # Create a temporary file to store the uploaded content
            uploaded_file = request.FILES.get('file')
            if not uploaded_file:
                return Response(
                    {"error": "No file provided"},
                    status=status.HTTP_400_BAD_REQUEST
                )

            # Generate a session ID if not provided
            session_id = request.data.get('session_id')
            if not session_id:
                session_id = uuid.uuid4().hex

            # Save the file to a temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as temp:
                for chunk in uploaded_file.chunks():
                    temp.write(chunk)
                temp_path = temp.name

            # Process the session document
            title = request.data.get('title', uploaded_file.name)
            chat_id = request.data.get('chat_id')

            # Create a new session document entry with minimal info
            session_doc = SessionDocument.objects.create(
                title=title,
                session_id=session_id,
                chat_id=chat_id,
                file_type=temp_path.split('.')[-1].lower(),
                uploaded_by=request.user,
                content_preview="Processing document..."  # Placeholder
            )

            # Start async processing
            process_id = uuid.uuid4().hex[:8]
            logger.info(f"Processing session document {session_doc.id} '{title}' asynchronously with ID: {process_id}")

            # Process in background thread - pass document_id to update existing document
            process_session_document_async(
                file_path=temp_path,
                session_id=session_id,
                chat_id=chat_id,
                title=title,
                user=request.user,
                document_id=session_doc.id  # Pass the document ID
            )

            # Return the document information immediately
            serializer = SessionDocumentSerializer(session_doc)
            response_data = serializer.data
            response_data['status'] = 'success'
            response_data['message'] = 'Document uploaded successfully and is being processed in the background.'

            return Response(
                response_data,
                status=status.HTTP_201_CREATED
            )

        except Exception as e:
            logger.error(f"Error processing session document: {e}")
            return Response(
                {"error": f"Error processing document: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class SessionDocumentDetailView(APIView):
    """
    API endpoint for retrieving and deleting session documents.
    """
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request, pk):
        """Get details for a specific session document."""
        try:
            # Add a short timeout to prevent hanging requests
            document = SessionDocument.objects.select_related('uploaded_by').get(pk=pk)

            # Get chunks with additional error handling
            try:
                # Limit chunk retrieval to avoid memory issues with large documents
                chunks = document.chunks.all()[:500]  # Limit to 500 chunks max
                chunk_count = chunks.count()

                # If too many chunks, provide a warning
                if chunk_count >= 500:
                    logger.warning(f"Document {pk} has more than 500 chunks, only returning first 500")
            except Exception as chunk_error:
                logger.error(f"Error retrieving chunks for document {pk}: {str(chunk_error)}")
                chunks = []
                chunk_count = 0

            # Use serializer with proper context and error handling
            try:
                serializer = SessionDocumentSerializer(document)
                return Response(serializer.data)
            except Exception as serialize_error:
                logger.error(f"Error serializing document {pk}: {str(serialize_error)}")
                # Return a simplified response if serialization fails
                return Response({
                    "id": document.id,
                    "title": document.title,
                    "session_id": document.session_id,
                    "chat_id": document.chat_id,
                    "content_preview": document.content_preview[:100] + "..." if document.content_preview else "",
                    "file_type": document.file_type,
                    "created_at": document.created_at,
                    "chunk_count": chunk_count,
                    "error": "Error retrieving full document details"
                })

        except SessionDocument.DoesNotExist:
            return Response(
                {"error": "Document not found"},
                status=status.HTTP_404_NOT_FOUND
            )
        except Exception as e:
            logger.error(f"Unexpected error in SessionDocumentDetailView.get: {str(e)}")
            return Response(
                {"error": f"Error retrieving document: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    def delete(self, request, pk):
        """Delete a session document."""
        try:
            document = SessionDocument.objects.get(pk=pk)
            document.delete()
            return Response(
                {"message": "Document deleted successfully"},
                status=status.HTTP_204_NO_CONTENT
            )
        except SessionDocument.DoesNotExist:
            return Response(
                {"error": "Document not found"},
                status=status.HTTP_404_NOT_FOUND
            )
        except Exception as e:
            logger.error(f"Error deleting document {pk}: {str(e)}")
            return Response(
                {"error": f"Error deleting document: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class SessionDocumentSearchView(APIView):
    """
    API endpoint for searching within session documents.
    """
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request):
        """Search within session documents using vector similarity."""
        # Validate request
        if not request.data.get('query') or not request.data.get('session_id'):
            return Response(
                {"error": "Both 'query' and 'session_id' are required"},
                status=status.HTTP_400_BAD_REQUEST
            )

        # Get parameters
        query = request.data.get('query')
        session_id = request.data.get('session_id')
        chat_id = request.data.get('chat_id')
        top_k = int(request.data.get('top_k', 3))

        # Perform the search
        results = vector_search_session_documents(
            query_text=query,
            session_id=session_id,
            chat_id=chat_id,
            top_k=top_k
        )

        return Response(results)


class CleanupSessionDocumentsView(APIView):
    """
    API endpoint for cleaning up session documents.
    """
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request):
        """Clean up session documents based on criteria."""
        session_id = request.data.get('session_id')
        chat_id = request.data.get('chat_id')
        older_than_days = request.data.get('older_than_days')

        # Require at least one filter
        if not any([session_id, chat_id, older_than_days]):
            return Response(
                {"error": "At least one filter (session_id, chat_id, older_than_days) is required"},
                status=status.HTTP_400_BAD_REQUEST
            )

        # Convert older_than_days to integer if provided
        if older_than_days:
            try:
                older_than_days = int(older_than_days)
            except ValueError:
                return Response(
                    {"error": "older_than_days must be an integer"},
                    status=status.HTTP_400_BAD_REQUEST
                )

        # Perform the cleanup
        deleted_count = cleanup_session_documents(
            session_id=session_id,
            chat_id=chat_id,
            older_than_days=older_than_days
        )

        return Response({
            "message": f"Successfully deleted {deleted_count} session documents",
            "deleted_count": deleted_count
        })