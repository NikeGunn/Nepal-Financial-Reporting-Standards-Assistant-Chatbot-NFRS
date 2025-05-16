"""
Document processing utilities for the NFRS Assistant.
"""
import fitz  # PyMuPDF
import os
import magic
import logging
import threading
from django.conf import settings
from utils.vector_ops import create_embedding, update_index_with_chunks
from django.db import transaction, IntegrityError, connection

logger = logging.getLogger(__name__)

# Add a process lock dictionary to prevent concurrent processing of the same document
_document_process_locks = {}
_process_lock = threading.Lock()

def extract_text_from_file(file_path):
    """
    Extract text from different file types.

    Args:
        file_path (str): Path to the file

    Returns:
        list: List of dictionaries with text and page information
    """
    mime_type = magic.from_file(file_path, mime=True)
    file_extension = os.path.splitext(file_path)[1].lower()

    if 'pdf' in mime_type or file_extension == '.pdf':
        return extract_text_from_pdf(file_path)

    elif 'text/plain' in mime_type or file_extension == '.txt':
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        return [{'text': text, 'page': 1}]

    elif 'docx' in mime_type or file_extension == '.docx':
        try:
            import docx
            doc = docx.Document(file_path)
            text = "\n".join([para.text for para in doc.paragraphs])
            return [{'text': text, 'page': 1}]
        except Exception as e:
            logger.error(f"Error extracting text from DOCX: {e}")
            return []

    else:
        logger.error(f"Unsupported file type: {mime_type}")
        return []


def extract_text_from_pdf(file_path):
    """
    Extract text from PDF file with page numbers.

    Args:
        file_path (str): Path to the PDF file

    Returns:
        list: List of dictionaries with text and page information
    """
    try:
        results = []
        pdf_document = fitz.open(file_path)

        for page_num, page in enumerate(pdf_document):
            text = page.get_text()
            if text.strip():
                results.append({
                    'text': text,
                    'page': page_num + 1
                })

        pdf_document.close()
        return results

    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        return []


def create_text_chunks(text, max_chunk_size=1000, overlap=100):
    """
    Split text into overlapping chunks.

    Args:
        text (str): Text to be split into chunks
        max_chunk_size (int): Maximum size of each chunk in characters
        overlap (int): Number of characters to overlap between chunks

    Returns:
        list: List of text chunks
    """
    chunks = []

    if len(text) <= max_chunk_size:
        chunks.append(text)
    else:
        start = 0
        while start < len(text):
            end = min(start + max_chunk_size, len(text))

            # Try to find a natural breaking point
            if end < len(text):
                # Look for good breaking points (periods, newlines)
                breaking_chars = ['. ', '.\n', '\n\n', '\n']
                best_break = end

                for char in breaking_chars:
                    # Look for the breaking character in the latter half of the current chunk
                    pos = text.rfind(char, start + max_chunk_size // 2, end)
                    if pos > start:
                        best_break = pos + len(char)
                        break

                end = best_break

            # Add the chunk
            chunks.append(text[start:end])

            # Move to next chunk with overlap
            start = max(start + 1, end - overlap)

    return chunks


def fast_process_document(document_id, max_pages=2, max_chunks=5):
    """
    Quickly process the first few pages of a document for immediate availability.

    This function rapidly extracts text from the first few pages, creates chunks,
    and generates embeddings for immediate use in the chat interface, while the
    full document continues processing in the background.

    Args:
        document_id: ID of the document to process
        max_pages: Maximum number of pages to process quickly
        max_chunks: Maximum number of chunks to process

    Returns:
        bool: Success status
    """
    from api.knowledge.models import DocumentChunk, Document
    import uuid  # Add import for UUID generation

    try:
        # Get the document
        document = Document.objects.get(id=document_id)

        # Update processing status to show partial progress
        document.processing_status = 'processing'
        document.save(update_fields=['processing_status'])

        # Extract text from the first few pages
        file_path = document.file.path
        mime_type = magic.from_file(file_path, mime=True)

        # For PDFs, extract only first few pages
        if 'pdf' in mime_type or file_path.lower().endswith('.pdf'):
            pdf_document = fitz.open(file_path)
            extracted_texts = []

            # Only process first few pages
            for page_num in range(min(max_pages, len(pdf_document))):
                page = pdf_document[page_num]
                text = page.get_text()
                if text.strip():
                    extracted_texts.append({
                        'text': text,
                        'page': page_num + 1
                    })

            pdf_document.close()
        else:
            # For other file types, just get beginning of text
            extracted_texts = extract_text_from_file(file_path)
            if len(extracted_texts) > 0:
                # For non-PDF files, just take the first part of the text
                text = extracted_texts[0]['text']
                # Limit text size
                limited_text = text[:max_chunks * 1000]  # Approximate size limit
                extracted_texts = [{'text': limited_text, 'page': 1}]

        # Process extracted text
        chunks_created = 0
        all_chunks = []

        for extracted in extracted_texts:
            text = extracted['text']
            page = extracted.get('page', 1)

            if not text.strip():
                continue

            # Split text into chunks
            chunks = create_text_chunks(text)

            # Create document chunks with embeddings
            for chunk_text in chunks:
                if not chunk_text.strip() or chunks_created >= max_chunks:
                    continue

                # Create embedding
                embedding = create_embedding(chunk_text)

                # Generate a unique chunk index using a combination of timestamp and random UUID
                # This ensures we don't conflict with regular chunk indexes or other fast-processed chunks
                unique_chunk_id = int.from_bytes(uuid.uuid4().bytes[:4], byteorder='big') + 1000000

                # Create chunk
                try:
                    doc_chunk = DocumentChunk(
                        document=document,
                        content=chunk_text,
                        chunk_index=unique_chunk_id,  # Use unique ID instead of sequential index
                        page_number=page
                    )

                    if embedding is not None:
                        doc_chunk.embedding_vector = embedding.tobytes()

                    doc_chunk.save()
                    all_chunks.append(doc_chunk)
                    chunks_created += 1
                except Exception as e:
                    # Log the error but continue processing other chunks
                    logger.error(f"Error saving fast-processed chunk for document {document_id}: {str(e)}")
                    continue

                # Break if we've reached the maximum number of chunks
                if chunks_created >= max_chunks:
                    break

            if chunks_created >= max_chunks:
                break

        # Update the vector index with these initial chunks
        if all_chunks:
            update_index_with_chunks(all_chunks)
            logger.info(f"Fast-processed document {document_id}: {chunks_created} chunks created for immediate use")
            return True

        return False

    except Exception as e:
        logger.error(f"Error in fast document processing: {e}")
        return False


def process_document(document_or_id):
    """
    Process a document: extract text, create chunks, generate embeddings.

    Args:
        document_or_id: Document model instance or document ID

    Returns:
        list: List of created DocumentChunk instances
    """
    from api.knowledge.models import DocumentChunk, Document
    from django.db import transaction, IntegrityError, connection
    import logging
    logger = logging.getLogger(__name__)

    try:
        # Check if the input is a document ID and retrieve the document
        document_id = None
        if isinstance(document_or_id, (int, str)):
            document_id = document_or_id
            try:
                document = Document.objects.select_for_update().get(id=document_id)
            except Document.DoesNotExist:
                logger.error(f"Document with ID {document_id} not found")
                return []
        else:
            document = document_or_id
            document_id = document.id

        # Use database transaction to ensure atomicity
        with transaction.atomic():
            # Requery to get the latest state with a lock
            document = Document.objects.select_for_update().get(id=document_id)

            # Check if document is already being processed by another thread
            if document.processing_status == 'processing':
                logger.warning(f"Document {document.id} is already being processed. Skipping.")
                return []

            # Update document status
            document.processing_status = 'processing'
            document.error_message = ''  # Clear any previous error messages
            document.save(update_fields=['processing_status', 'error_message'])

            # CRITICAL: Force delete existing chunks using raw SQL to avoid Django ORM transaction issues
            # This ensures the deletion is complete before we start adding new chunks
            try:
                cursor = connection.cursor()
                # Delete regular chunks but keep fast-processed ones (with high chunk_index)
                cursor.execute("DELETE FROM knowledge_documentchunk WHERE document_id = %s AND chunk_index < 10000", [document.id])
                deleted_count = cursor.rowcount
                logger.info(f"Successfully deleted {deleted_count} existing chunks for document {document.id} using raw SQL")
                # Close cursor to release resources
                cursor.close()
            except Exception as delete_error:
                logger.error(f"Error deleting existing chunks for document {document.id}: {delete_error}")
                # If deletion fails, mark the document as failed and exit
                document.processing_status = 'failed'
                document.error_message = f"Failed to prepare document for processing: {str(delete_error)}"
                document.save(update_fields=['processing_status', 'error_message'])
                return []

        # After the transaction completes, process the document

        # Extract text from file
        file_path = document.file.path
        extracted_texts = extract_text_from_file(file_path)

        if not extracted_texts:
            document.processing_status = 'failed'
            document.error_message = "Failed to extract text from document"
            document.save(update_fields=['processing_status', 'error_message'])
            return []

        # Process each extracted text (usually one per page for PDFs)
        created_chunks = []
        chunk_index = 0

        # Process each extracted text (usually one per page for PDFs)
        for extracted in extracted_texts:
            text = extracted['text']
            page = extracted.get('page', 1)

            if not text.strip():
                continue

            # Split text into chunks
            chunks = create_text_chunks(text)

            # Create document chunks with embeddings
            for chunk_text in chunks:
                # Skip empty chunks
                if not chunk_text.strip():
                    continue

                # Create embedding vector
                try:
                    embedding = create_embedding(chunk_text)

                    # Handle potential integrity errors with retries
                    max_retries = 3
                    retry_count = 0
                    success = False

                    while not success and retry_count < max_retries:
                        try:
                            # Create document chunk
                            doc_chunk = DocumentChunk(
                                document=document,
                                content=chunk_text,
                                chunk_index=chunk_index,
                                page_number=page
                            )

                            if embedding is not None:
                                doc_chunk.embedding_vector = embedding.tobytes()

                            doc_chunk.save()
                            created_chunks.append(doc_chunk)
                            chunk_index += 1
                            success = True

                        except IntegrityError as integrity_error:
                            # If we hit a unique constraint, try the next index
                            logger.warning(f"Integrity error for chunk {chunk_index}, trying next index: {integrity_error}")
                            chunk_index += 1
                            retry_count += 1

                        except Exception as save_error:
                            logger.error(f"Error saving chunk {chunk_index}: {save_error}")
                            chunk_index += 1
                            break

                except Exception as chunk_error:
                    logger.error(f"Error creating chunk {chunk_index} for document {document.id}: {chunk_error}")
                    # Continue with the next chunk rather than failing the whole process
                    chunk_index += 1  # Still increment the index to avoid duplicates

        # Update document status
        if created_chunks:
            document.processing_status = 'completed'
            logger.info(f"Successfully processed document {document.id} with {len(created_chunks)} chunks")

            # Delete any fast-processed chunks now that full processing is done
            try:
                DocumentChunk.objects.filter(document_id=document.id, chunk_index__gte=10000).delete()
                logger.info(f"Removed temporary fast-processed chunks for document {document.id}")
            except Exception as cleanup_error:
                logger.error(f"Error cleaning up fast-processed chunks: {cleanup_error}")

            # Update the vector index with the new chunks
            update_index_with_chunks(created_chunks)
        else:
            document.processing_status = 'failed'
            document.error_message = "No valid text chunks could be extracted"
            logger.error(f"Failed to create any chunks for document {document.id}")

        document.save(update_fields=['processing_status', 'error_message'])
        return created_chunks

    except Exception as e:
        logger.error(f"Error processing document: {e}")
        try:
            # Get the document if we only have the ID
            if isinstance(document_or_id, (int, str)):
                document = Document.objects.get(id=document_or_id)
            document.processing_status = 'failed'
            document.error_message = str(e)[:500]  # Limit error message length
            document.save(update_fields=['processing_status', 'error_message'])
        except Exception as nested_e:
            logger.error(f"Failed to update document status after error: {nested_e}")
        return []


# Function to handle document processing in a background thread
def process_document_async(document_id):
    """
    Process a document asynchronously in a separate thread.
    Ensures only one thread processes a specific document at a time.

    Args:
        document_id: ID of the document to process
    """
    # First do quick processing for immediate availability
    try:
        # Skip fast processing if it's already been processed to avoid the constraint error
        from api.knowledge.models import DocumentChunk
        if not DocumentChunk.objects.filter(document_id=document_id).exists():
            fast_process_document(document_id)
        else:
            logger.info(f"Document {document_id} has existing chunks. Skipping fast processing.")
    except Exception as e:
        logger.error(f"Error in fast document processing: {e}")

    # Then start full processing in background
    # Check if this document is already being processed
    with _process_lock:
        if document_id in _document_process_locks and _document_process_locks[document_id].is_alive():
            logger.warning(f"Document {document_id} is already being processed in another thread. Skipping.")
            return None

    def _worker():
        try:
            logger.info(f"Starting background processing for document {document_id}")
            process_document(document_id)
            logger.info(f"Completed background processing for document {document_id}")
        except Exception as e:
            logger.error(f"Error in async document processing for document {document_id}: {str(e)}")
        finally:
            # Clean up the lock when done
            with _process_lock:
                if document_id in _document_process_locks:
                    del _document_process_locks[document_id]

    # Start processing in a daemon thread to avoid blocking
    thread = threading.Thread(target=_worker)
    thread.daemon = True

    # Register this thread in our locks dictionary
    with _process_lock:
        _document_process_locks[document_id] = thread

    thread.start()
    return thread


# Function to process browser session documents
def process_session_document(file_path, session_id, chat_id=None, title=None, user=None):
    """
    Process a document uploaded within a browser session, without storing in the database.
    This is for temporary document handling that's tied to a specific chat session.

    Args:
        file_path (str): Path to the temporary document file
        session_id (str): Browser session identifier
        chat_id (str, optional): Chat ID if the document is tied to a specific conversation
        title (str, optional): Document title
        user (User, optional): User who uploaded the document

    Returns:
        SessionDocument: The created session document or None if processing failed
    """
    from api.knowledge.models import SessionDocument, SessionDocumentChunk
    import uuid

    try:
        # Extract file type
        file_extension = os.path.splitext(file_path)[1].lower().replace('.', '')
        mime_type = magic.from_file(file_path, mime=True)

        if file_extension not in ['pdf', 'txt', 'docx']:
            if 'pdf' in mime_type:
                file_extension = 'pdf'
            elif 'text/plain' in mime_type:
                file_extension = 'txt'
            elif 'docx' in mime_type or 'word' in mime_type:
                file_extension = 'docx'
            else:
                logger.error(f"Unsupported file type for session document: {mime_type}")
                return None

        # Generate a title if not provided
        if not title:
            title = os.path.basename(file_path)

        # Extract text from the document
        extracted_texts = extract_text_from_file(file_path)
        if not extracted_texts:
            logger.error(f"Failed to extract text from session document: {file_path}")
            return None

        # Create a content preview
        content_preview = ""
        for text_data in extracted_texts[:2]:  # Just use first couple of pages for preview
            content_preview += text_data['text'][:250]
            if len(content_preview) >= 250:
                content_preview = content_preview[:250] + "..."
                break

        # Create the session document
        session_doc = SessionDocument.objects.create(
            title=title,
            content_preview=content_preview,
            session_id=session_id,
            chat_id=chat_id,
            file_type=file_extension,
            uploaded_by=user
        )

        # Process each extracted text into chunks with embeddings
        chunks_created = 0

        for idx, extracted in enumerate(extracted_texts):
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

                # Generate embedding
                try:
                    embedding = create_embedding(chunk_text)

                    # Create the session document chunk
                    chunk = SessionDocumentChunk(
                        session_document=session_doc,
                        content=chunk_text,
                        chunk_index=chunks_created,
                        page_number=page
                    )

                    if embedding is not None:
                        chunk.embedding_vector = embedding.tobytes()

                    chunk.save()
                    chunks_created += 1

                except Exception as chunk_error:
                    logger.error(f"Error processing session document chunk: {str(chunk_error)}")
                    # Continue with other chunks
                    continue

        logger.info(f"Successfully processed session document: {session_doc.id} with {chunks_created} chunks")
        return session_doc

    except Exception as e:
        logger.error(f"Error processing session document: {str(e)}")
        return None


def vector_search_session_documents(query_text, session_id, chat_id=None, top_k=3):
    """
    Search for relevant content in session documents using vector similarity.

    Args:
        query_text (str): The query text to search for
        session_id (str): Browser session identifier
        chat_id (str, optional): Specific chat ID to filter documents
        top_k (int): Maximum number of results to return

    Returns:
        list: List of relevant document chunks with similarity scores
    """
    from api.knowledge.models import SessionDocumentChunk
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity

    try:
        # Generate query embedding
        query_embedding = create_embedding(query_text)
        if query_embedding is None:
            logger.error("Failed to create query embedding")
            return []

        # Find all session document chunks for this session
        filter_kwargs = {"session_document__session_id": session_id}
        if chat_id:
            filter_kwargs["session_document__chat_id"] = chat_id

        chunks = SessionDocumentChunk.objects.filter(
            **filter_kwargs,
            embedding_vector__isnull=False
        )

        if not chunks:
            return []

        # Calculate similarities
        results = []
        for chunk in chunks:
            chunk_embedding = np.frombuffer(chunk.embedding_vector, dtype=np.float32)
            similarity = cosine_similarity(
                [query_embedding],
                [chunk_embedding]
            )[0][0]

            results.append({
                'chunk_id': chunk.id,
                'document_id': chunk.session_document_id,
                'document_title': chunk.session_document.title,
                'content': chunk.content,
                'page_number': chunk.page_number,
                'similarity': float(similarity)
            })

        # Sort by similarity and return top_k results
        sorted_results = sorted(results, key=lambda x: x['similarity'], reverse=True)
        return sorted_results[:top_k]

    except Exception as e:
        logger.error(f"Error in session document vector search: {str(e)}")
        return []


def cleanup_session_documents(session_id=None, chat_id=None, older_than_days=None):
    """
    Clean up temporary session documents.

    Args:
        session_id (str, optional): Clean up documents for a specific session
        chat_id (str, optional): Clean up documents for a specific chat
        older_than_days (int, optional): Clean up documents older than specified days

    Returns:
        int: Number of documents deleted
    """
    from api.knowledge.models import SessionDocument, SessionDocumentChunk
    from django.utils import timezone
    from django.db import transaction, connection
    from datetime import timedelta

    try:
        # Build filter criteria
        filter_kwargs = {}
        if session_id:
            filter_kwargs['session_id'] = session_id
        if chat_id:
            filter_kwargs['chat_id'] = chat_id
        if older_than_days:
            cutoff_date = timezone.now() - timedelta(days=older_than_days)
            filter_kwargs['created_at__lt'] = cutoff_date

        # Use a transaction to ensure atomicity
        with transaction.atomic():
            # First, get IDs of documents to be deleted
            doc_ids_to_delete = list(SessionDocument.objects.filter(**filter_kwargs).values_list('id', flat=True))
            
            if not doc_ids_to_delete:
                logger.info(f"No session documents found to delete with filters: {filter_kwargs}")
                return 0
                
            # Use raw SQL to delete chunks first to handle potential race conditions
            if doc_ids_to_delete:
                try:
                    cursor = connection.cursor()
                    # Convert list to string for SQL IN clause
                    doc_ids_str = ','.join(map(str, doc_ids_to_delete))
                    # Delete all chunks associated with these documents
                    cursor.execute(f"DELETE FROM knowledge_sessiondocumentchunk WHERE session_document_id IN ({doc_ids_str})")
                    chunks_deleted = cursor.rowcount
                    cursor.close()
                    logger.info(f"Deleted {chunks_deleted} session document chunks for documents {doc_ids_to_delete}")
                except Exception as e:
                    logger.error(f"Error deleting session document chunks: {e}")
            
            # Now delete the documents
            deleted_count = SessionDocument.objects.filter(id__in=doc_ids_to_delete).delete()[0]
            logger.info(f"Deleted {deleted_count} session documents with IDs: {doc_ids_to_delete}")
            
            return deleted_count

    except Exception as e:
        logger.error(f"Error cleaning up session documents: {str(e)}")
        return 0