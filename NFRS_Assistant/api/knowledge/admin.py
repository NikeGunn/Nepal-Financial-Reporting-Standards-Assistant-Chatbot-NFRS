from django.contrib import admin
from django.urls import path
from django.shortcuts import render, redirect
from django.http import HttpResponse, HttpResponseRedirect
from django.contrib import messages
from django.utils.html import format_html
from django.urls import reverse
from django.core.exceptions import PermissionDenied
import logging
import threading

from .models import Document, DocumentChunk, VectorIndex

logger = logging.getLogger(__name__)

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

    def get_urls(self):
        urls = super().get_urls()
        custom_urls = [
            path('upload-document/', self.admin_site.admin_view(self.upload_document_view),
                 name='knowledge_document_upload'),
        ]
        return custom_urls + urls

    def add_view(self, request, form_url='', extra_context=None):
        """Override the default add view to redirect to our custom upload form."""
        return HttpResponseRedirect(reverse('admin:knowledge_document_upload'))

    def upload_document_view(self, request):
        """Custom view for admin document upload."""
        try:
            if not request.user.is_staff:
                raise PermissionDenied("You don't have permission to upload documents.")

            if request.method == 'POST':
                from .serializers import DocumentUploadSerializer
                from django.db import transaction
                from django.http import JsonResponse

                # Create serializer with request data
                # Combine POST data and FILES into a single data dictionary
                data = request.POST.copy()
                data.update(request.FILES)
                serializer = DocumentUploadSerializer(data=data, context={'request': request})

                if serializer.is_valid():
                    # Save document with admin upload type
                    document = serializer.save()
                    document.upload_type = 'admin'
                    document.processing_status = 'pending'
                    document.save()

                    # Start background processing
                    def process_document_async(doc_id):
                        """Run document processing in a separate thread."""
                        from utils.document_processor import process_document

                        def run_processing():
                            try:
                                # Process the document
                                process_document(doc_id)
                            except Exception as e:
                                logger.error(f"Error in background processing thread: {e}")
                                # Try to update document status on error
                                try:
                                    doc = Document.objects.get(id=doc_id)
                                    doc.processing_status = 'failed'
                                    doc.error_message = str(e)[:500]  # Limit error message length
                                    doc.save()
                                except Exception as nested_error:
                                    logger.error(f"Failed to update document status after error: {nested_error}")

                        # Start background thread and detach it
                        thread = threading.Thread(target=run_processing)
                        thread.daemon = True  # Thread will terminate when main program exits
                        thread.start()

                    transaction.on_commit(lambda: process_document_async(document.id))

                    # Show success message
                    messages.success(request, f'Document "{document.title}" uploaded successfully and is being processed.')

                    # For AJAX requests, return JSON response
                    if request.headers.get('x-requested-with') == 'XMLHttpRequest':
                        return JsonResponse({
                            'status': 'success',
                            'message': f'Document "{document.title}" uploaded successfully and is being processed.',
                            'redirect_url': reverse('admin:knowledge_document_changelist')
                        })

                    # For normal requests, redirect to document list
                    return HttpResponseRedirect(reverse('admin:knowledge_document_changelist'))
                else:
                    # Show error message
                    for field, errors in serializer.errors.items():
                        for error in errors:
                            messages.error(request, f"{field}: {error}")

                    # For AJAX requests, return JSON response with errors
                    if request.headers.get('x-requested-with') == 'XMLHttpRequest':
                        return JsonResponse({
                            'status': 'error',
                            'errors': serializer.errors
                        }, status=400)

            # Use try/except specifically for template rendering
            try:
                # Render upload form with the correct template path
                context = {
                    'title': 'Upload Knowledge Base Document',
                    'document_types': Document.DOCUMENT_TYPES,
                    'opts': self.model._meta,
                    'app_label': self.model._meta.app_label,
                }
                return render(request, 'admin/knowledge/document/upload_form.html', context)
            except Exception as template_error:
                logger.error(f"Template rendering error: {str(template_error)}")
                # Fall back to the standard admin add form if the template cannot be found
                return super(DocumentAdmin, self).add_view(request)

        except Exception as e:
            logger.error(f"Error in admin document upload: {str(e)}")
            messages.error(request, f"Error uploading document: {str(e)}")

            # For AJAX requests, return JSON response with error
            if request.headers.get('x-requested-with') == 'XMLHttpRequest':
                return JsonResponse({
                    'status': 'error',
                    'message': f"Error uploading document: {str(e)}"
                }, status=500)

            return HttpResponseRedirect(reverse('admin:knowledge_document_changelist'))

    def changelist_view(self, request, extra_context=None):
        extra_context = extra_context or {}
        extra_context['upload_url'] = reverse('admin:knowledge_document_upload')
        return super().changelist_view(request, extra_context=extra_context)


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