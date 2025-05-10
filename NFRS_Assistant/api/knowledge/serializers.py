from rest_framework import serializers
from .models import Document, DocumentChunk, VectorIndex


class DocumentChunkSerializer(serializers.ModelSerializer):
    """
    Serializer for document chunks.
    """
    class Meta:
        model = DocumentChunk
        fields = ['id', 'document', 'content', 'chunk_index', 'page_number', 'created_at']
        read_only_fields = ['id', 'created_at']


class DocumentSerializer(serializers.ModelSerializer):
    """
    Serializer for documents.
    """
    file_type_display = serializers.SerializerMethodField()
    uploaded_by_username = serializers.SerializerMethodField()
    processing_status_display = serializers.SerializerMethodField()
    chunk_count = serializers.SerializerMethodField()

    class Meta:
        model = Document
        fields = [
            'id', 'title', 'description', 'file', 'file_type', 'file_type_display',
            'upload_type', 'uploaded_by', 'uploaded_by_username', 'is_public',
            'processing_status', 'processing_status_display', 'error_message',
            'created_at', 'updated_at', 'chunk_count'
        ]
        read_only_fields = ['id', 'uploaded_by', 'processing_status', 'error_message', 'created_at', 'updated_at']

    def get_file_type_display(self, obj):
        return dict(Document.DOCUMENT_TYPES).get(obj.file_type, obj.file_type)

    def get_uploaded_by_username(self, obj):
        return obj.uploaded_by.username

    def get_processing_status_display(self, obj):
        return dict(Document.PROCESSING_STATUS).get(obj.processing_status, obj.processing_status)

    def get_chunk_count(self, obj):
        return obj.chunks.count()


class DocumentUploadSerializer(serializers.ModelSerializer):
    """
    Serializer for document uploads.
    """
    class Meta:
        model = Document
        fields = ['title', 'description', 'file', 'file_type', 'is_public']

    def validate_file(self, value):
        # Validate file type and size
        file_extension = value.name.split('.')[-1].lower()
        if file_extension not in ['pdf', 'txt', 'docx']:
            raise serializers.ValidationError("Unsupported file type. Please upload PDF, TXT, or DOCX files.")

        # Check file size (10MB limit)
        if value.size > 10 * 1024 * 1024:
            raise serializers.ValidationError("File is too large. Maximum size is 10MB.")

        return value

    def validate_file_type(self, value):
        # Ensure file_type matches the actual file extension
        file_extension = self.initial_data.get('file').name.split('.')[-1].lower()
        if value != file_extension:
            raise serializers.ValidationError(f"File type should be '{file_extension}' based on the uploaded file.")
        return value

    def create(self, validated_data):
        # Set uploaded_by to current user
        validated_data['uploaded_by'] = self.context['request'].user

        # Set upload type based on user role
        profile = self.context['request'].user.profile
        validated_data['upload_type'] = 'admin' if profile.is_admin else 'user'

        return super().create(validated_data)


class VectorIndexSerializer(serializers.ModelSerializer):
    """
    Serializer for vector indices.
    """
    class Meta:
        model = VectorIndex
        fields = ['id', 'name', 'description', 'index_file_path', 'dimension', 'num_vectors', 'last_updated', 'is_active']
        read_only_fields = ['id', 'index_file_path', 'dimension', 'num_vectors', 'last_updated']


class SearchQuerySerializer(serializers.Serializer):
    """
    Serializer for vector search queries.
    """
    query = serializers.CharField(required=True)
    top_k = serializers.IntegerField(required=False, default=5, min_value=1, max_value=20)
    filter_document_ids = serializers.ListField(
        child=serializers.IntegerField(),
        required=False,
        allow_empty=True
    )