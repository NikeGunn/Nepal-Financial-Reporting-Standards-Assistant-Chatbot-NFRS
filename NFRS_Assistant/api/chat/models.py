from django.db import models
from django.contrib.auth.models import User

class Conversation(models.Model):
    """
    Model to store conversation sessions between users and the chatbot.
    """
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='conversations')
    title = models.CharField(max_length=255, blank=True)
    language = models.CharField(max_length=5, choices=[('en', 'English'), ('ne', 'Nepali')], default='en')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    is_active = models.BooleanField(default=True)

    class Meta:
        ordering = ['-updated_at']

    def __str__(self):
        return f"{self.title or 'Untitled'} - {self.user.username}"

    def save(self, *args, **kwargs):
        # Auto-generate title from first message if not provided
        if not self.title and hasattr(self, 'messages') and self.messages.exists():
            first_message = self.messages.first().content
            self.title = first_message[:50] + ('...' if len(first_message) > 50 else '')
        super().save(*args, **kwargs)


class Message(models.Model):
    """
    Model to store individual messages within a conversation.
    """
    ROLE_CHOICES = [
        ('user', 'User'),
        ('assistant', 'Assistant'),
        ('system', 'System'),
    ]

    conversation = models.ForeignKey(Conversation, on_delete=models.CASCADE, related_name='messages')
    role = models.CharField(max_length=10, choices=ROLE_CHOICES)
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    # References to knowledge sources used to generate this message
    knowledge_sources = models.ManyToManyField('knowledge.Document', related_name='referenced_in_messages', blank=True)

    class Meta:
        ordering = ['created_at']

    def __str__(self):
        return f"{self.role}: {self.content[:50]}..."