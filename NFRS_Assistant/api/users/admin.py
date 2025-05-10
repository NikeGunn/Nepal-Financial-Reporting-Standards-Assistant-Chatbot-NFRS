from django.contrib import admin
from .models import UserProfile, ApiKey


@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = ('user', 'preferred_language', 'organization', 'is_admin', 'created_at')
    list_filter = ('is_admin', 'preferred_language')
    search_fields = ('user__username', 'user__email', 'organization')
    raw_id_fields = ('user',)
    date_hierarchy = 'created_at'


@admin.register(ApiKey)
class ApiKeyAdmin(admin.ModelAdmin):
    list_display = ('name', 'user', 'is_active', 'created_at', 'last_used_at')
    list_filter = ('is_active',)
    search_fields = ('name', 'user__username', 'key')
    raw_id_fields = ('user',)
    readonly_fields = ('key', 'created_at', 'last_used_at')
    date_hierarchy = 'created_at'