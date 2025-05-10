"""
Background task handler for NFRS Assistant.

This script can be run as a separate process to handle background tasks:
- Document processing
- Vector index updates
- Scheduled maintenance tasks

Run with: python manage.py shell < tasks.py
Or set up as a scheduled job with cron/supervisor.
"""
import os
import sys
import django
import logging
import time
from datetime import datetime, timedelta

# Set up Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'NFRS_Assistant.settings.dev')
django.setup()

# Import models and utilities after Django setup
from api.knowledge.models import Document, VectorIndex
from utils.document_processor import process_document
from utils.vector_ops import update_index_with_chunks

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('background_tasks.log')
    ]
)
logger = logging.getLogger(__name__)


def process_pending_documents():
    """Process documents in 'pending' status."""
    pending_docs = Document.objects.filter(processing_status='pending')
    logger.info(f"Found {pending_docs.count()} pending documents to process")

    for doc in pending_docs:
        logger.info(f"Processing document: {doc.id} - {doc.title}")
        process_document(doc)


def update_vector_indices():
    """Update vector indices with new document chunks."""
    logger.info("Updating vector indices")
    update_index_with_chunks()


def clean_up_old_indices():
    """Clean up old vector indices that are no longer active."""
    # Get indices that are inactive and older than 7 days
    cutoff_date = datetime.now() - timedelta(days=7)
    old_indices = VectorIndex.objects.filter(
        is_active=False,
        last_updated__lt=cutoff_date
    )

    for index in old_indices:
        logger.info(f"Cleaning up old index: {index.name}")

        # Delete the index file
        if os.path.exists(index.index_file_path):
            try:
                os.remove(index.index_file_path)

                # Delete metadata file
                metadata_path = os.path.join(os.path.dirname(index.index_file_path),
                                             "index_metadata.json")
                if os.path.exists(metadata_path):
                    os.remove(metadata_path)

                logger.info(f"Deleted index files for {index.name}")
            except Exception as e:
                logger.error(f"Error deleting index file: {e}")

        # Delete the index record
        index.delete()


def run_tasks():
    """Run all background tasks."""
    logger.info("Starting background tasks")
    start_time = time.time()

    try:
        # Process pending documents
        process_pending_documents()

        # Update vector indices
        update_vector_indices()

        # Clean up old indices
        clean_up_old_indices()

    except Exception as e:
        logger.error(f"Error in background tasks: {e}")

    elapsed_time = time.time() - start_time
    logger.info(f"Background tasks completed in {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    run_tasks()