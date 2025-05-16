from rest_framework import serializers
from django.contrib.auth.models import User
from .models import Conversation, Message
from api.knowledge.models import Document

class UserBasicSerializer(serializers.ModelSerializer):
    """
    Simple serializer for basic user information.
    """
    class Meta:
        model = User
        fields = ['id', 'username', 'first_name', 'last_name', 'email']
        read_only_fields = fields


class MessageSerializer(serializers.ModelSerializer):
    """
    Serializer for Message model.
    """
    knowledge_sources = serializers.PrimaryKeyRelatedField(
        queryset=Document.objects.all(),
        many=True,
        required=False
    )

    class Meta:
        model = Message
        fields = ['id', 'conversation', 'role', 'content', 'created_at', 'knowledge_sources']
        read_only_fields = ['id', 'created_at']


class ConversationListSerializer(serializers.ModelSerializer):
    """
    Serializer for listing conversations.
    """
    message_count = serializers.SerializerMethodField()
    user = UserBasicSerializer(read_only=True)
    last_message = serializers.SerializerMethodField()
    last_activity = serializers.SerializerMethodField()

    class Meta:
        model = Conversation
        fields = ['id', 'title', 'language', 'created_at', 'updated_at', 'is_active',
                  'message_count', 'user', 'last_message', 'last_activity']
        read_only_fields = ['id', 'created_at', 'updated_at']

    def get_message_count(self, obj):
        return obj.messages.count()

    def get_last_message(self, obj):
        """Return a preview of the last message in the conversation."""
        last_message = obj.messages.order_by('-created_at').first()
        if last_message:
            return {
                'id': last_message.id,
                'role': last_message.role,
                'content_preview': last_message.content[:100] + ('...' if len(last_message.content) > 100 else ''),
                'created_at': last_message.created_at
            }
        return None

    def get_last_activity(self, obj):
        """Return the timestamp of the last activity in this conversation."""
        return obj.updated_at


class ConversationDetailSerializer(serializers.ModelSerializer):
    """
    Serializer for conversation details including messages.
    """
    messages = MessageSerializer(many=True, read_only=True)
    user = UserBasicSerializer(read_only=True)

    class Meta:
        model = Conversation
        fields = ['id', 'title', 'language', 'created_at', 'updated_at', 'is_active', 'user', 'messages']
        read_only_fields = ['id', 'created_at', 'updated_at']


class ChatMessageSerializer(serializers.Serializer):
    """
    Serializer for handling incoming chat messages.
    """
    message = serializers.CharField(required=True)
    conversation_id = serializers.IntegerField(required=False, allow_null=True)
    language = serializers.CharField(required=False, default='en')
    use_multi_agent = serializers.BooleanField(required=False, default=True)
    session_id = serializers.CharField(required=False, allow_null=True)

    def validate_message(self, value):
        """Validate that the message isn't empty."""
        if not value.strip():
            raise serializers.ValidationError("Message cannot be empty")
        return value

    def validate_language(self, value):
        """Validate that the language is supported."""
        supported_languages = ['en', 'ne']  # English and Nepali
        if value not in supported_languages:
            raise serializers.ValidationError(
                f"Language '{value}' is not supported. Supported languages: {', '.join(supported_languages)}"
            )
        return value


class TranslateMessageSerializer(serializers.Serializer):
    """
    Serializer for translating messages between languages.
    """
    message_id = serializers.IntegerField(required=True)
    target_language = serializers.CharField(required=True)