from django.db import models
from django.contrib.auth.models import User

class Document(models.Model):
    """
    Model to store uploaded documents and their metadata.
    """
    DOCUMENT_TYPES = [
        ('pdf', 'PDF Document'),
        ('txt', 'Text Document'),
        ('docx', 'Word Document'),
    ]

    UPLOAD_TYPES = [
        ('user', 'User Upload'),
        ('admin', 'Admin Upload'),
    ]

    PROCESSING_STATUS = [
        ('pending', 'Pending'),
        ('processing', 'Processing'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]

    title = models.CharField(max_length=255)
    description = models.TextField(blank=True)
    file = models.FileField(upload_to='documents/')
    file_type = models.CharField(max_length=10, choices=DOCUMENT_TYPES)
    upload_type = models.CharField(max_length=10, choices=UPLOAD_TYPES)
    uploaded_by = models.ForeignKey(User, on_delete=models.CASCADE, related_name='uploaded_documents')
    is_public = models.BooleanField(default=False)
    processing_status = models.CharField(max_length=20, choices=PROCESSING_STATUS, default='pending')
    error_message = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return self.title


class DocumentChunk(models.Model):
    """
    Model to store chunked text from documents for embedding and retrieval.
    """
    document = models.ForeignKey(Document, on_delete=models.CASCADE, related_name='chunks')
    content = models.TextField()
    chunk_index = models.IntegerField()
    embedding_vector = models.BinaryField(null=True, blank=True)  # Store FAISS vectors as binary
    page_number = models.IntegerField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['document', 'chunk_index']
        unique_together = ['document', 'chunk_index']

    def __str__(self):
        return f"{self.document.title} - Chunk {self.chunk_index}"


class VectorIndex(models.Model):
    """
    Model to store metadata about FAISS vector indices.
    """
    name = models.CharField(max_length=255, unique=True)
    description = models.TextField(blank=True)
    index_file_path = models.CharField(max_length=255)
    dimension = models.IntegerField(default=1536)  # Default for OpenAI embeddings
    num_vectors = models.IntegerField(default=0)
    last_updated = models.DateTimeField(auto_now=True)
    is_active = models.BooleanField(default=True)

    def __str__(self):
        return self.name