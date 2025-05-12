"""
Document processing utilities for the NFRS Assistant.
"""
import fitz  # PyMuPDF
import os
import magic
import logging
import threading
from django.conf import settings
from utils.vector_ops import create_embedding

logger = logging.getLogger(__name__)


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


def process_document(document_or_id):
    """
    Process a document: extract text, create chunks, generate embeddings.

    Args:
        document_or_id: Document model instance or document ID

    Returns:
        list: List of created DocumentChunk instances
    """
    from api.knowledge.models import DocumentChunk, Document
    from django.db import transaction
    import logging
    logger = logging.getLogger(__name__)

    try:
        # Check if the input is a document ID and retrieve the document
        if isinstance(document_or_id, (int, str)):
            try:
                document = Document.objects.get(id=document_or_id)
            except Document.DoesNotExist:
                logger.error(f"Document with ID {document_or_id} not found")
                return []
        else:
            document = document_or_id

        # Check if document is already being processed by another thread
        if document.processing_status == 'processing':
            logger.warning(f"Document {document.id} is already being processed. Skipping.")
            return []

        # Update document status
        document.processing_status = 'processing'
        document.error_message = '' # Clear any previous error messages
        document.save(update_fields=['processing_status', 'error_message'])

        # Use a transaction for cleaning up existing chunks to ensure atomicity
        with transaction.atomic():
            # First clean up any existing chunks to avoid unique constraint violations
            existing_chunks_count = DocumentChunk.objects.filter(document=document).count()
            if existing_chunks_count > 0:
                logger.info(f"Removing {existing_chunks_count} existing chunks for document {document.id}")
                DocumentChunk.objects.filter(document=document).delete()
                # Force a commit to ensure chunks are actually deleted before proceeding
                transaction.commit()
                logger.info(f"Successfully deleted all existing chunks for document {document.id}")

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
                embedding = create_embedding(chunk_text)

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
                except Exception as chunk_error:
                    logger.error(f"Error creating chunk {chunk_index} for document {document.id}: {chunk_error}")
                    # Continue with the next chunk rather than failing the whole process

        # Update document status
        if created_chunks:
            document.processing_status = 'completed'
            logger.info(f"Successfully processed document {document.id} with {len(created_chunks)} chunks")
        else:
            document.processing_status = 'failed'
            document.error_message = "No valid text chunks could be extracted"
            logger.error(f"Failed to create any chunks for document {document.id}")

        document.save(update_fields=['processing_status', 'error_message'])
        return created_chunks

    except Exception as e:
        logger.error(f"Error processing document: {e}")
        try:
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

    Args:
        document_id: ID of the document to process
    """
    def _worker():
        try:
            logger.info(f"Starting background processing for document {document_id}")
            process_document(document_id)
            logger.info(f"Completed background processing for document {document_id}")
        except Exception as e:
            logger.error(f"Error in async document processing for document {document_id}: {str(e)}")

    # Start processing in a daemon thread to avoid blocking
    thread = threading.Thread(target=_worker)
    thread.daemon = True
    thread.start()
    return thread