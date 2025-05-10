"""
Custom permissions for the NFRS Assistant API.
"""
from rest_framework import permissions


class IsAdminUser(permissions.BasePermission):
    """
    Custom permission to only allow admin users to access a view.
    """

    def has_permission(self, request, view):
        # Check if user is authenticated and has admin role in profile
        return (
            request.user
            and request.user.is_authenticated
            and hasattr(request.user, 'profile')
            and request.user.profile.is_admin
        )


class IsDocumentOwnerOrPublic(permissions.BasePermission):
    """
    Custom permission to only allow owners of a document to edit it
    or anyone to view if document is public.
    """

    def has_object_permission(self, request, view, obj):
        # Read permissions are allowed to any request for public documents
        if request.method in permissions.SAFE_METHODS and obj.is_public:
            return True

        # Write permissions are only allowed to the document owner
        return obj.uploaded_by == request.user


class IsConversationOwner(permissions.BasePermission):
    """
    Custom permission to only allow owners of a conversation to access it.
    """

    def has_object_permission(self, request, view, obj):
        # Permissions are only allowed to the conversation owner
        return obj.user == request.user


class IsMessageOwner(permissions.BasePermission):
    """
    Custom permission to only allow owners of a message to access it.
    """

    def has_object_permission(self, request, view, obj):
        # Permissions are only allowed to the message owner (via conversation)
        return obj.conversation.user == request.user


class IsProfileOwner(permissions.BasePermission):
    """
    Custom permission to only allow users to access their own profile.
    """

    def has_object_permission(self, request, view, obj):
        # Permissions are only allowed to the profile owner
        return obj.user == request.user