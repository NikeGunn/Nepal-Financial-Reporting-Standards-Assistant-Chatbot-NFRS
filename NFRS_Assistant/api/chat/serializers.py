from rest_framework import serializers
from .models import Conversation, Message
from api.knowledge.models import Document

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

    class Meta:
        model = Conversation
        fields = ['id', 'title', 'language', 'created_at', 'updated_at', 'is_active', 'message_count']
        read_only_fields = ['id', 'created_at', 'updated_at']

    def get_message_count(self, obj):
        return obj.messages.count()


class ConversationDetailSerializer(serializers.ModelSerializer):
    """
    Serializer for conversation details including messages.
    """
    messages = MessageSerializer(many=True, read_only=True)

    class Meta:
        model = Conversation
        fields = ['id', 'title', 'language', 'created_at', 'updated_at', 'is_active', 'messages']
        read_only_fields = ['id', 'created_at', 'updated_at']


class ChatMessageSerializer(serializers.Serializer):
    """
    Serializer for handling incoming chat messages.
    """
    message = serializers.CharField(required=True)
    conversation_id = serializers.IntegerField(required=False, allow_null=True)
    language = serializers.CharField(required=False, default='en')


class TranslateMessageSerializer(serializers.Serializer):
    """
    Serializer for translating messages between languages.
    """
    message_id = serializers.IntegerField(required=True)
    target_language = serializers.CharField(required=True)