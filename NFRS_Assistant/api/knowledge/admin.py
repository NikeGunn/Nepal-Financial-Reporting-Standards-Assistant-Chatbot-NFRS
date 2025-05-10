from django.contrib import admin
from .models import Document, DocumentChunk, VectorIndex


class DocumentChunkInline(admin.TabularInline):
    model = DocumentChunk
    extra = 0
    readonly_fields = ('created_at', 'embedding_vector')
    fields = ('chunk_index', 'page_number', 'content', 'created_at')
    can_delete = False
    max_num = 10  # Limit the number of chunks shown


@admin.register(Document)
class DocumentAdmin(admin.ModelAdmin):
    list_display = ('id', 'title', 'file_type', 'upload_type', 'uploaded_by',
                    'processing_status', 'is_public', 'created_at')
    list_filter = ('file_type', 'upload_type', 'processing_status', 'is_public')
    search_fields = ('title', 'description', 'uploaded_by__username')
    readonly_fields = ('created_at', 'updated_at', 'processing_status', 'error_message')
    date_hierarchy = 'created_at'
    inlines = [DocumentChunkInline]


@admin.register(DocumentChunk)
class DocumentChunkAdmin(admin.ModelAdmin):
    list_display = ('id', 'document', 'chunk_index', 'page_number', 'get_content_preview', 'created_at')
    list_filter = ('document__file_type', 'created_at')
    search_fields = ('content', 'document__title')
    readonly_fields = ('created_at', 'embedding_vector')
    raw_id_fields = ('document',)
    date_hierarchy = 'created_at'

    def get_content_preview(self, obj):
        return obj.content[:100] + ('...' if len(obj.content) > 100 else '')
    get_content_preview.short_description = 'Content Preview'


@admin.register(VectorIndex)
class VectorIndexAdmin(admin.ModelAdmin):
    list_display = ('id', 'name', 'dimension', 'num_vectors', 'is_active', 'last_updated')
    list_filter = ('is_active', 'dimension')
    search_fields = ('name', 'description')
    readonly_fields = ('last_updated', 'num_vectors')