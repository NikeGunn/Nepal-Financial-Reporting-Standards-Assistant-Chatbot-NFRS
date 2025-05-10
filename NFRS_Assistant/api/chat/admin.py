from django.contrib import admin
from .models import Conversation, Message


class MessageInline(admin.TabularInline):
    model = Message
    extra = 0
    readonly_fields = ('created_at',)
    fields = ('role', 'content', 'created_at')
    can_delete = False
    max_num = 20  # Limit the number of messages shown


@admin.register(Conversation)
class ConversationAdmin(admin.ModelAdmin):
    list_display = ('id', 'user', 'title', 'language', 'created_at', 'updated_at', 'is_active')
    list_filter = ('language', 'is_active')
    search_fields = ('title', 'user__username')
    readonly_fields = ('created_at', 'updated_at')
    date_hierarchy = 'created_at'
    inlines = [MessageInline]


@admin.register(Message)
class MessageAdmin(admin.ModelAdmin):
    list_display = ('id', 'get_conversation_title', 'role', 'get_content_preview', 'created_at')
    list_filter = ('role', 'created_at')
    search_fields = ('content', 'conversation__title', 'conversation__user__username')
    readonly_fields = ('created_at',)
    raw_id_fields = ('conversation', 'knowledge_sources')
    date_hierarchy = 'created_at'

    def get_conversation_title(self, obj):
        return obj.conversation.title or f"Conversation {obj.conversation.id}"
    get_conversation_title.short_description = 'Conversation'

    def get_content_preview(self, obj):
        return obj.content[:100] + ('...' if len(obj.content) > 100 else '')
    get_content_preview.short_description = 'Content'