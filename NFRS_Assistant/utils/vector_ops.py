"""
Vector operations utilities for the NFRS Assistant.
"""
import os
import numpy as np
import json
import openai
from django.conf import settings
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from api.knowledge.models import DocumentChunk, VectorIndex, Document
import logging

logger = logging.getLogger(__name__)


def create_embedding(text):
    """
    Create vector embedding for text using OpenAI API.

    Args:
        text (str): The text to embed

    Returns:
        numpy.ndarray: The embedding vector
    """
    try:
        # Import OpenAI client
        from openai import OpenAI

        # Get API key from settings
        api_key = settings.OPENAI_API_KEY

        if not api_key:
            logger.error("OpenAI API key is not configured")
            raise ValueError("OpenAI API key is missing. Please configure it in your environment variables.")

        # Initialize the client with the API key
        client = OpenAI(api_key=api_key)

        # Make the API call with proper configuration
        response = client.embeddings.create(
            model=settings.EMBEDDING_MODEL or "text-embedding-3-small",
            input=text
        )

        # Extract embedding from response
        embedding = np.array(response.data[0].embedding, dtype=np.float32)
        return embedding
    except Exception as e:
        logger.error(f"Error creating embedding: {e}")
        print(f"Error creating embedding: {e}")
        raise


def get_active_index():
    """
    Get the currently active vector index.

    Returns:
        tuple: (list, str, list) - The embeddings list, its path, and metadata
    """
    try:
        # Get active vector index from database
        vector_index = VectorIndex.objects.filter(is_active=True).first()
        if not vector_index:
            logger.warning("No active vector index found")
            return None, None, None

        # Check if index file exists with the exact path
        index_path = vector_index.index_file_path

        # If the exact path doesn't exist, try to find it relative to the project's vector_store directory
        if not os.path.exists(index_path):
            logger.warning(f"Vector index file not found at original path: {index_path}")

            # Try to locate the file in the project's vector_store directory
            filename = os.path.basename(index_path)
            alternative_path = os.path.join(settings.VECTOR_STORE_DIR, filename)

            if os.path.exists(alternative_path):
                logger.info(f"Found vector index at alternative path: {alternative_path}")
                index_path = alternative_path

                # Update the database with the correct path
                vector_index.index_file_path = alternative_path
                vector_index.save()
            else:
                logger.warning(f"Vector index file not found at alternative path: {alternative_path}")
                return None, None, None

        # Load scikit-learn compatible index
        with open(index_path, 'rb') as f:
            embeddings = pickle.load(f)

        # Try multiple potential metadata paths
        base_filename = os.path.splitext(os.path.basename(index_path))[0]
        metadata_paths = [
            # Original path - filename based on the index file name
            os.path.join(os.path.dirname(index_path), f"{base_filename}_metadata.json"),
            # Common path attempted in the error message
            os.path.join(os.path.dirname(index_path), "index_metadata.json"),
            # Any specific files that might exist in vector_store directory
            os.path.join(os.path.dirname(index_path), "nfrs_index_82806537_metadata.json")
        ]

        metadata = None
        for path in metadata_paths:
            if os.path.exists(path):
                logger.info(f"Vector index metadata found at: {path}")
                with open(path, 'r') as f:
                    metadata = json.load(f)
                break

        if metadata is None:
            # If no metadata file found but we have embeddings, create an empty metadata array
            # This allows the system to function with reduced capabilities
            logger.warning(f"No vector index metadata found at any expected location. Creating empty metadata.")
            metadata = []

        return embeddings, index_path, metadata

    except Exception as e:
        logger.error(f"Error loading vector index: {e}")
        return None, None, None


def vector_search(query, top_k=5, filter_document_ids=None):
    """
    Perform vector search to find relevant document chunks.

    Args:
        query (str): The search query
        top_k (int): Number of top results to return
        filter_document_ids (list): Optional list of document IDs to filter results

    Returns:
        list: List of dictionaries with search results
    """
    try:
        # Get embedding for query
        query_embedding = create_embedding(query)
        if query_embedding is None:
            logger.error("Failed to create embedding for query")
            return []

        # Reshape for scikit-learn
        query_embedding = query_embedding.reshape(1, -1)

        # Get active index and metadata
        embeddings, _, metadata = get_active_index()
        if embeddings is None or metadata is None or len(embeddings) == 0:
            logger.error("Failed to load active index or index is empty")
            return []

        # Calculate similarities using scikit-learn
        similarities = cosine_similarity(query_embedding, embeddings)[0]

        # Get indices of top-k similarities
        indices = np.argsort(similarities)[::-1][:top_k * 2]  # Get more for filtering

        # Process and filter results
        results = []
        for idx in indices:
            if idx < len(metadata):
                # Apply document filter if provided
                doc_id = metadata[idx]['document_id']
                if filter_document_ids and doc_id not in filter_document_ids:
                    continue

                # Add to results
                results.append({
                    'score': float(similarities[idx]),
                    'document_id': doc_id,
                    'chunk_id': metadata[idx]['chunk_id'],
                    'content': metadata[idx].get('content', ''),
                    'page_number': metadata[idx].get('page_number')
                })

                # Stop if we have enough results
                if len(results) >= top_k:
                    break

        return results

    except Exception as e:
        logger.error(f"Vector search error: {e}")
        return []


def update_index_with_chunks(chunks=None):
    """
    Update the vector index with new document chunks.

    Args:
        chunks (QuerySet, optional): DocumentChunk queryset to add to the index

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Get active vector index
        vector_index = VectorIndex.objects.filter(is_active=True).first()
        index_path = None

        if vector_index:
            index_path = vector_index.index_file_path

        # Load existing index or create new one
        if index_path and os.path.exists(index_path):
            try:
                with open(index_path, 'rb') as f:
                    embeddings = pickle.load(f)
            except:
                embeddings = []

            # Load existing metadata
            metadata_path = os.path.join(os.path.dirname(index_path), "index_metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            else:
                metadata = []
        else:
            # Create a new index
            embeddings = []
            metadata = []

            # Create a new index record if needed
            if not vector_index:
                # Ensure vector store directory exists
                os.makedirs(settings.VECTOR_STORE_DIR, exist_ok=True)

                # Generate unique name for the index
                import uuid
                index_name = f"nfrs_index_{uuid.uuid4().hex[:8]}"
                index_path = os.path.join(settings.VECTOR_STORE_DIR, f"{index_name}.index")

                # Create new vector index record
                vector_index = VectorIndex.objects.create(
                    name=index_name,
                    description="NFRS documents vector index",
                    index_file_path=index_path,
                    is_active=True
                )

        # Get chunks to add to the index
        if chunks is None:
            # Get all chunks with embeddings that aren't in the index yet
            existing_chunk_ids = [meta.get('chunk_id') for meta in metadata if 'chunk_id' in meta]
            chunks = DocumentChunk.objects.filter(
                embedding_vector__isnull=False
            ).exclude(id__in=existing_chunk_ids)

        # Add chunks to the index
        if hasattr(chunks, 'exists') and chunks.exists():
            new_embeddings = []
            new_metadata = []

            for chunk in chunks:
                # Convert binary data to numpy array
                vector = np.frombuffer(chunk.embedding_vector, dtype=np.float32)
                new_embeddings.append(vector)

                # Add metadata for the chunk
                new_metadata.append({
                    'chunk_id': chunk.id,
                    'document_id': chunk.document_id,
                    'content': chunk.content[:500],  # Store preview of content
                    'page_number': chunk.page_number
                })

            # Add vectors to the index
            if new_embeddings:
                embeddings.extend(new_embeddings)

                # Update metadata
                metadata.extend(new_metadata)

                # Save updated index and metadata
                metadata_path = os.path.join(os.path.dirname(index_path), "index_metadata.json")

                with open(index_path, 'wb') as f:
                    pickle.dump(np.array(embeddings), f)

                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f)

                # Update index record
                vector_index.num_vectors = len(metadata)
                vector_index.save()

                logger.info(f"Vector index updated with {len(new_embeddings)} new vectors")
                return True

        return True

    except Exception as e:
        logger.error(f"Error updating vector index: {e}")
        return False


def search_documents(query, top_k=5, filter_document_ids=None):
    """
    Search for documents relevant to a query.

    Args:
        query (str): The search query
        top_k (int): Number of top results to return
        filter_document_ids (list): Optional list of document IDs to filter results

    Returns:
        list: List of search results
    """
    # Use the existing vector_search function to find relevant documents
    return vector_search(query, top_k, filter_document_ids)
