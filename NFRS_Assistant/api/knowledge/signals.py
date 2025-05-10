from django.db.models.signals import post_save
from django.dispatch import receiver
from .models import Document, DocumentChunk
from django.db import transaction
from utils.document_processor import process_document
from utils.vector_ops import update_index_with_chunks
import logging

logger = logging.getLogger(__name__)


@receiver(post_save, sender=Document)
def process_new_document(sender, instance, created, **kwargs):
    """
    Signal to process a newly uploaded document.

    This will extract text, create chunks, and generate embeddings for the document.
    The processing runs in a separate transaction to avoid blocking the request.
    """
    if created or (instance.processing_status == 'pending' and not kwargs.get('raw', False)):
        # Skip if this is a raw save (like from a fixture) or if already processing
        if kwargs.get('raw', False) or instance.processing_status != 'pending':
            return

        # Use transaction.on_commit to process the document after the current transaction completes
        transaction.on_commit(lambda: process_document_async(instance.id))


def process_document_async(document_id):
    """
    Process a document asynchronously.

    This function is designed to be called after the current transaction completes.
    """
    try:
        # Get the document - this is in a new transaction
        document = Document.objects.get(id=document_id)

        # Skip if document is already being processed or completed
        if document.processing_status != 'pending':
            return

        # Process the document (extract text, create chunks, generate embeddings)
        logger.info(f"Processing document: {document.id} - {document.title}")
        chunks = process_document(document)

        # Update the vector index with the new chunks
        if chunks:
            logger.info(f"Updating vector index with {len(chunks)} new chunks")
            update_index_with_chunks(chunks)

    except Exception as e:
        logger.error(f"Error in async document processing: {e}")

        # Update document status to failed
        try:
            document = Document.objects.get(id=document_id)
            document.processing_status = 'failed'
            document.error_message = str(e)
            document.save(update_fields=['processing_status', 'error_message'])
        except:
            pass  # If we can't update the document, we've already logged the error


@receiver(post_save, sender=DocumentChunk)
def update_vector_index(sender, instance, created, **kwargs):
    """
    Signal to update vector index when a new document chunk is created.

    This is a backup for cases where the main document processing signal fails.
    """
    # Only process newly created chunks with embedding vectors
    if created and instance.embedding_vector and not kwargs.get('raw', False):
        # Use transaction.on_commit to update the index after the current transaction completes
        transaction.on_commit(lambda: update_vector_index_async(instance.id))


def update_vector_index_async(chunk_id):
    """
    Update vector index asynchronously for a single chunk.
    """
    try:
        # Get the chunk - this is in a new transaction
        chunk = DocumentChunk.objects.filter(id=chunk_id).first()
        if chunk and chunk.embedding_vector:
            # Update the vector index with just this chunk
            logger.info(f"Updating vector index with chunk: {chunk.id}")
            update_index_with_chunks([chunk])
    except Exception as e:
        logger.error(f"Error in async vector index update: {e}")