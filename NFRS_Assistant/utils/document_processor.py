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
        chunk_index = 0
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

                # Create chunk
                doc_chunk = DocumentChunk(
                    document=document,
                    content=chunk_text,
                    chunk_index=10000 + chunk_index,  # Use high index to avoid conflicts with full processing
                    page_number=page
                )

                if embedding is not None:
                    doc_chunk.embedding_vector = embedding.tobytes()

                doc_chunk.save()
                all_chunks.append(doc_chunk)
                chunks_created += 1
                chunk_index += 1

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
        fast_process_document(document_id)
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